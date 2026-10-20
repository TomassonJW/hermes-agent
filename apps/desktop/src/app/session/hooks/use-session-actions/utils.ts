import { resolveSessionRpcOwner } from '@/app/contrib/wiring-routing'
import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { getSession } from '@/hermes'
import {
  assistantTextPart,
  type ChatMessage,
  chatMessageText,
  preserveLocalAssistantErrors,
  textPart,
  toChatMessages
} from '@/lib/chat-messages'
import { normalizePersonalityValue } from '@/lib/chat-runtime'
import { embeddedImageUrls, textWithoutEmbeddedImages } from '@/lib/embedded-images'
import { parseErrorSurface } from '@/lib/error-surface'
import { isMessagingSource, normalizeSessionSource } from '@/lib/session-source'
import { isLiveTailReplyId } from '@/lib/spoken-reply'
import { reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { requestDesktopOnboardingForCredentialWarning } from '@/store/onboarding'
import { pendingSnapshotFence, reconcilePendingSubmissions } from '@/store/pending-submissions'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $projectTree } from '@/store/projects'
import {
  $cronSessions,
  $currentCwd,
  $messagingSessions,
  $sessions,
  commitWorkspaceCwdForSelectedSession,
  getSessionOwnerHint,
  knownSessionOwner,
  ownerLookupSessionRows,
  releaseWorkspaceCwdOwner,
  sessionMatchesStoredId,
  setCronSessions,
  setCurrentBranch,
  setCurrentCwdTransient,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentModelTransient,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentProviderTransient,
  setCurrentReasoningEffort,
  setCurrentReasoningEffortWire,
  setCurrentServiceTier,
  setCurrentUsage,
  setMessagingSessions,
  setSessionOwnerHint,
  setSessions,
  setUnlistedSessionOwnerRows,
  setWorkspaceCwdOwner,
  setYoloActive
} from '@/store/session'
import {
  $removedSessionIds,
  captureSessionTombstoneGenerations,
  type SessionTombstoneGenerationSnapshot,
  tombstoneLifecycleChanged
} from '@/store/session-removal'
import type { SessionProfileRoute } from '@/store/session-request-router'
import { runtimeSessionOwner, sessionTileOwnerRoute } from '@/store/session-states'

// Re-exported for the many session-actions/tile call sites that already import
// it from here; the canonical definition lives in @/store/session.
export { sessionMatchesStoredId }
import { sessionOwnerRouteFromRow, type SessionOwnerScope } from '@/store/session-request-router'
import { reportBackendContract, reportInstallMethodWarning } from '@/store/updates'
import type { SessionCreateResponse, SessionInfo, SessionResumeResult, SessionRuntimeInfo } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

import {
  carryRowIdentity,
  hasStructuralParts,
  isLiveTailRow,
  isStrictAnswerTextExtension,
  localPendingSupersedes,
  withAuthoritativeTurnState
} from './pending-turn-content'
import { conflictingTranscriptIdentity, persistedTurnsEquivalent, transcriptRowIds } from './pending-turn-identity'
import { preserveLocalPendingTurnMessages } from './pending-turn-preservation'

export { isStrictAnswerTextExtension } from './pending-turn-content'
export { preserveLocalPendingTurnMessages } from './pending-turn-preservation'

export { goneSessionVerdict, isSessionGoneError, resolveResumedBusy } from './resume-verdicts'

function withAppendedText(message: ChatMessage, suffix: string): ChatMessage {
  let appended = false

  const parts = message.parts.map(part => {
    if (part.type !== 'text' || appended) {
      return part
    }

    appended = true

    return { ...part, text: `${part.text}${suffix}` }
  })

  return appended ? { ...message, parts } : message
}

/**
 * Whether a committed bubble already holds the reply a stale live snapshot was
 * streaming. History folds a tool turn's narration, tools and answer into one
 * bubble while the snapshot carries the turn's whole text, so compare the
 * bubble's text with whitespace collapsed. The bubble must end in a reply: a
 * mid-turn commit ends on a tool call and the turn is still running.
 */
function committedReplyCovers(message: ChatMessage, snapshot: string): boolean {
  const text = message.parts
    .flatMap(part => (part.type === 'text' ? [part.text] : []))
    .join(' ')
    .replace(/\s+/g, ' ')

  return Boolean(replyAfterLastTool(message).trim()) && isStrictAnswerTextExtension(text, snapshot.replace(/\s+/g, ' '))
}

/** A folded tool-turn bubble's answer: its text after the last tool call. */
function replyAfterLastTool(message: ChatMessage): string {
  const lastTool = message.parts.findLastIndex(part => part.type === 'tool-call')

  return chatMessageText({ ...message, parts: message.parts.slice(lastTool + 1) })
}

/**
 * True only when a row was written after the live turn began. Text alone
 * cannot tell this turn's reply from the previous turn's answer to the same
 * resent prompt, so a missing timestamp (older runtime) never passes.
 */
function committedDuringTurn(turnStartedAt: unknown, committedAt: unknown): boolean {
  return typeof turnStartedAt === 'number' && typeof committedAt === 'number' && committedAt >= turnStartedAt
}

/**
 * Carry structural parts an authoritative row cannot express.
 *
 * A live turn's authoritative projection is TEXT-ONLY: the gateway's `inflight`
 * snapshot carries `user`/`assistant` strings, and history is not committed
 * until the turn finishes. The renderer's cached state is therefore the sole
 * carrier of the running turn's reasoning and tool calls, so switching threads
 * mid-turn and back re-hydrated an assistant row stripped of both — the turn
 * looked inert, with no thinking trace and no tool activity.
 *
 * Preserved only when the rows are the SAME turn: identical text, or the
 * authoritative text extending the cached one (another delta landed). Anything
 * else may be a different turn at the same role ordinal — compression rewrites
 * history — and must not inherit foreign parts. Tool calls dedupe on
 * `toolCallId` so a row that already carries them is left alone.
 */
function preserveStructuralParts(message: ChatMessage, previous: ChatMessage): ChatMessage {
  const carried = previous.parts.filter(part => part.type === 'reasoning' || part.type === 'tool-call')

  if (!carried.length) {
    return message
  }

  const hasReasoning = message.parts.some(part => part.type === 'reasoning')

  const presentToolCallIds = new Set(
    message.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))
  )

  const missing = carried.filter(part =>
    part.type === 'reasoning' ? !hasReasoning : !presentToolCallIds.has(part.toolCallId)
  )

  return missing.length ? { ...message, parts: [...missing, ...message.parts] } : message
}

// Compile-time exhaustiveness guards. If a new field is added to ChatMessage
// or a new part type appears in the ChatMessagePart union (e.g. @assistant-ui
// ships one), these fail tsc until someone explicitly classifies it.
//
// COMPARED: fields whose change must trigger a re-render (setMessages).
// Durable identity changes must publish too: view-side refresh reconciliation
// must see an acknowledgement even when its visible text is unchanged.
// IGNORED:  fields that are intentionally not compared — display-only metadata
//           or reference identity the runtime already guarantees.
//   attachmentRefs — composer-side metadata; already reconciled in reconcileResumeMessages
//   serverRowSpan — backend rows the folded message covers; the older-page offset
//                   accounting reads it, the transcript never paints it
//
// If your new field affects what the user sees in the transcript, add it to
// COMPARED. If it's metadata that shouldn't trigger a re-render, add it to
// IGNORED.
const _chatMessageFieldsExhaustive: {
  [K in Exclude<keyof ChatMessage, (typeof COMPARED_FIELDS)[number] | (typeof IGNORED_FIELDS)[number]>]: never
} = {}

const COMPARED_FIELDS = [
  'rowId',
  'persistedTurn',
  'durableComplete',
  'recovered',
  'asyncResult',
  'asyncResultKind',
  'id',
  'role',
  'pending',
  'error',
  // Structured failure layer — drives the error card's title and action row,
  // so a change (e.g. resume replay attaching the descriptor) must repaint.
  'errorSurface',
  'hidden',
  'branchGroupId',
  'interim',
  'interrupted',
  'reactions',
  'timestamp',
  'completedAt',
  // Turn wall-clock duration — stamps the visible "⏱ 38s" badge, so a change
  // must re-render (set once at completion; stable afterwards).
  'durationS'
] as const

const IGNORED_FIELDS = ['attachmentRefs', 'parts', 'serverRowSpan'] as const

// Compile-time check: every ChatMessagePart discriminant must be handled by
// chatPartsEquivalent. If @assistant-ui adds a new part type, this fails tsc.
//   text, reasoning      → compared by .text
//   tool-call             → compared by toolCallId/toolName + result presence
//   source, image, file, data, generative-ui, audio, data-* → shallow primitive compare
const _chatMessagePartTypesExhaustive: {
  [T in Exclude<ChatMessage['parts'][number]['type'], (typeof HANDLED_PART_TYPES)[number]>]: never
} = {}

const HANDLED_PART_TYPES = [
  'text',
  'reasoning',
  'tool-call',
  'source',
  'image',
  'file',
  'data',
  'generative-ui',
  'audio'
] as const

// Structural compare WITHOUT JSON.stringify — the only consumer asks "did
// the transcript change, should I call setMessages?", so a slightly
// conservative compare (occasionally false-negative → one extra idempotent
// setMessages) is safe, but a false-POSITIVE (claiming equal when different)
// would skip a needed update.
export function chatPartsEquivalent(aPart: ChatMessage['parts'][number], bPart: ChatMessage['parts'][number]): boolean {
  // Reference equality fast-path
  if (aPart === bPart) {
    return true
  }

  if (aPart.type !== bPart.type) {
    return false
  }

  if (aPart.timestamp !== bPart.timestamp || aPart.completedAt !== bPart.completedAt) {
    return false
  }

  if (aPart.type === 'text' || aPart.type === 'reasoning') {
    if (aPart.sourceRowId !== bPart.sourceRowId) {
      return false
    }

    return (aPart as { text: string }).text === (bPart as { text: string }).text
  }

  if (aPart.type === 'tool-call') {
    const aCall = aPart as { toolCallId?: string; toolName?: string; result?: unknown }
    const bCall = bPart as { toolCallId?: string; toolName?: string; result?: unknown }

    if (aCall.toolCallId !== bCall.toolCallId || aCall.toolName !== bCall.toolName) {
      return false
    }

    // Compare whether result is present (undefined on both or defined on both)
    const aHasResult = aCall.result !== undefined
    const bHasResult = bCall.result !== undefined

    return aHasResult === bHasResult
  }

  // For all other handled part types (source, image, file, data, generative-ui,
  // audio, data-*), fall back to shallow primitive-key comparison — conservative:
  // if we're not sure, claim not-equal (one extra setMessages is harmless, but
  // skipping an update would break the UI).
  // SAFETY: both parts already narrowed by `type` to flat primitive part kinds;
  // the cast only exposes their keys for comparison, never reinterprets them.
  const aPrimitive = aPart as unknown as Record<string, unknown>
  // SAFETY: same narrowing as above; key comparison only.
  const bPrimitive = bPart as unknown as Record<string, unknown>
  const aKeys = Object.keys(aPrimitive).filter(k => typeof aPrimitive[k] !== 'object' || aPrimitive[k] === null)
  const bKeys = Object.keys(bPrimitive).filter(k => typeof bPrimitive[k] !== 'object' || bPrimitive[k] === null)

  if (aKeys.length !== bKeys.length) {
    return false
  }

  return aKeys.every(k => aPrimitive[k] === bPrimitive[k])
}

export function chatReactionsEquivalent(a: ChatMessage['reactions'], b: ChatMessage['reactions']): boolean {
  const aList = a ?? []
  const bList = b ?? []

  if (aList === bList) {
    return true
  }

  return (
    aList.length === bList.length &&
    aList.every((reaction, index) => reaction.emoji === bList[index].emoji && reaction.author === bList[index].author)
  )
}

export function chatMessagesEquivalent(a: ChatMessage, b: ChatMessage): boolean {
  if (
    a.id !== b.id ||
    a.rowId !== b.rowId ||
    !persistedTurnsEquivalent(a.persistedTurn, b.persistedTurn) ||
    a.role !== b.role ||
    a.durableComplete !== b.durableComplete ||
    a.recovered !== b.recovered ||
    a.pending !== b.pending ||
    a.error !== b.error ||
    // Structural compare — the descriptor arrives as a fresh object per
    // resume/replay, so identity comparison would repaint forever.
    (a.errorSurface?.layer ?? null) !== (b.errorSurface?.layer ?? null) ||
    (a.errorSurface?.code ?? null) !== (b.errorSurface?.code ?? null) ||
    (a.errorSurface?.retryable ?? null) !== (b.errorSurface?.retryable ?? null) ||
    a.hidden !== b.hidden ||
    a.branchGroupId !== b.branchGroupId ||
    a.timestamp !== b.timestamp ||
    a.completedAt !== b.completedAt ||
    // Interim gates the action footer, so flipping it must repaint (e.g. a
    // previewed final settling onto a sealed interim bubble restores the bar).
    (a.interim ?? false) !== (b.interim ?? false) ||
    !chatReactionsEquivalent(a.reactions, b.reactions)
  ) {
    return false
  }

  if (a.parts.length !== b.parts.length) {
    return false
  }

  return a.parts.every((part, index) => chatPartsEquivalent(part, b.parts[index]))
}

export function chatMessageArraysEquivalent(a: ChatMessage[], b: ChatMessage[]): boolean {
  // Array-level identity fast-path (same reference)
  if (a === b) {
    return true
  }

  return a.length === b.length && a.every((message, index) => chatMessagesEquivalent(message, b[index]))
}

/**
 * Keep the CURRENT array when the replacement is content-equivalent.
 *
 * The resume reconcilers create fresh `ChatMessage` objects via
 * `toChatMessages` even when nothing changed. Publishing those unconditionally
 * replaces the `$messages`/session-slice array with a new reference of fresh
 * objects — and because `useRuntimeMessageRepository` keys its normalization
 * cache (and React keys its rows) by object identity, every message in the
 * window re-normalizes and remounts: full markdown re-parse + shiki
 * re-highlight per row, on the main thread, per warm session switch (#95595).
 * Returning `current` when the content is equivalent keeps array AND object
 * identity, so the warm switch is O(1) paint.
 */
export function preserveEquivalentTranscript(current: ChatMessage[], next: ChatMessage[]): ChatMessage[] {
  return chatMessageArraysEquivalent(current, next) ? current : next
}

/** Durable history against the local view: role-ordinal pairing, then the
 *  local pending turn and local assistant errors the DB cannot know about. */
export function reconcileDurableHistory(messages: ChatMessage[], previous: ChatMessage[]): ChatMessage[] {
  const reconciled = reconcileResumeMessages(messages, previous)
  const withPendingTurn = preserveLocalPendingTurnMessages(reconciled, previous)

  return preserveLocalAssistantErrors(withPendingTurn, previous)
}

export function reconcileResumeMessages(nextMessages: ChatMessage[], previousMessages: ChatMessage[]): ChatMessage[] {
  if (!previousMessages.length) {
    return nextMessages
  }

  const previousByRoleOrdinal = new Map<string, ChatMessage>()
  const previousRoleCounts = new Map<string, number>()

  for (const message of previousMessages) {
    const ordinal = previousRoleCounts.get(message.role) ?? 0
    previousRoleCounts.set(message.role, ordinal + 1)
    previousByRoleOrdinal.set(`${message.role}:${ordinal}`, message)
  }

  const nextRoleCounts = new Map<string, number>()

  return nextMessages.map(message => {
    const ordinal = nextRoleCounts.get(message.role) ?? 0
    nextRoleCounts.set(message.role, ordinal + 1)

    const previous = previousByRoleOrdinal.get(`${message.role}:${ordinal}`)

    if (!previous || conflictingTranscriptIdentity(previous, message)) {
      return message
    }

    const nextText = chatMessageText(message).trim()
    const previousText = chatMessageText(previous)
    const previousVisibleText = textWithoutEmbeddedImages(previousText)
    const previousTrimmed = previousVisibleText.trim()
    let preserved = message

    // #75825: resume can project an empty (or lagging) inflight assistant shell
    // at the same role-ordinal as the live stream row that still holds the
    // streamed text, reasoning and tool calls. Prefer that richer pending row
    // instead of painting the shell — otherwise the reply vanishes until
    // restart. Guarded to the same reply further along (see
    // localPendingSupersedes) so a different turn at the same ordinal cannot
    // hijack the slot.
    if (localPendingSupersedes(previous, message)) {
      return withAuthoritativeTurnState(previous, message)
    }

    // Empty prose carries no identity: an empty-text cached assistant and an
    // empty/tool-only hydrated row can share a role ordinal while being
    // different turns, and plain `'' === ''` would pair them (#114543),
    // grafting the cached reasoning/tool parts onto the unrelated row.
    const sameText = nextText.length > 0 && (nextText === previousVisibleText || nextText === previousText.trim())

    // Mid-turn, the authoritative text has advanced past the cached copy by one
    // or more deltas. That is still the same turn, and the cached row holds the
    // only copy of its reasoning / tool calls, so treat an extension as a match
    // for structural carry-over. Attachment refs and image re-appending stay on
    // the strict equality path — they reconcile a SETTLED row, and a growing
    // row is by definition not settled.
    //
    // Live-tail identity: structure-only same-turn carry is allowed only when
    // the *structure-bearing cached row* is still the in-flight stream
    // (pending / stream id / interim). Marking only the text-only next row
    // live is not enough — after compression a new live assistant can share a
    // role ordinal with an unrelated historical structured row and must not
    // inherit its reasoning/tool parts (#76444 review / salvage).
    const sameTurn =
      sameText ||
      (nextText.length > 0 && previousTrimmed.length > 0 && isStrictAnswerTextExtension(nextText, previousTrimmed)) ||
      (message.role === 'assistant' &&
        previous.role === 'assistant' &&
        hasStructuralParts(previous) &&
        !hasStructuralParts(message) &&
        isLiveTailRow(previous) &&
        isLiveTailRow(message))

    if (sameTurn) {
      preserved = preserveStructuralParts(preserved, previous)

      // Never replace structured answer text with a non-extending flat dump.
      if (
        message.role === 'assistant' &&
        hasStructuralParts(previous) &&
        !hasStructuralParts(message) &&
        !isStrictAnswerTextExtension(nextText, previousVisibleText)
      ) {
        const nonText = preserved.parts.filter(part => part.type !== 'text')
        const priorAnswer = previous.parts.filter(part => part.type === 'text')
        preserved = { ...preserved, parts: [...nonText, ...priorAnswer] }
      }
    }

    if (
      sameText &&
      message.role === 'user' &&
      preserved.attachmentRefs === undefined &&
      previous.attachmentRefs?.length
    ) {
      preserved = { ...preserved, attachmentRefs: [...previous.attachmentRefs] }
    }

    // Reactions and the row id come from the same authoritative rows as the
    // text, but a live/optimistic row that hasn't round-tripped yet carries
    // neither. Carry the cached copy forward so a reaction doesn't blink off
    // mid-turn. NEW object every time — the runtime repository's WeakMap
    // caches normalized ThreadMessages by ChatMessage identity.
    if (sameTurn) {
      preserved = carryRowIdentity(preserved, previous)
    }

    const previousImages = embeddedImageUrls(previousText)

    if (!previousImages.length || embeddedImageUrls(chatMessageText(preserved)).length) {
      return preserved
    }

    if (nextText !== previousVisibleText) {
      return preserved
    }

    return withAppendedText(preserved, previousImages.map(url => `\n${url}`).join(''))
  })
}

/**
 * Append the backend-only tail of a live turn to a stored transcript.
 *
 * Session history is committed only when a turn finishes. During a reconnect,
 * `inflight` is therefore the authority for the currently running user/assistant
 * pair, while `queued` is an accepted next-turn prompt waiting in gateway
 * memory. Stable ids let repeated activate/resume hydration reconcile instead
 * of growing duplicate rows.
 */
const safelyPersistedInflightUser = Symbol('safelyPersistedInflightUser')

type LiveSessionProjection = Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id' | 'turn_started_at'> & {
  [safelyPersistedInflightUser]?: true
}

type ReconciledSessionResumeResult = SessionResumeResult & {
  [safelyPersistedInflightUser]?: true
}

export function appendLiveSessionProjection(messages: ChatMessage[], projection: LiveSessionProjection): ChatMessage[] {
  const inflightUser = projection.inflight?.user?.trim() ?? ''
  const inflightAssistant = projection.inflight?.assistant ?? ''
  const inflightStreaming = Boolean(projection.inflight?.streaming)

  // Mid-turn redirect corrections. They are additional user bubbles belonging
  // to this same turn, ordered by arrival: after the output that had already
  // streamed when they were typed, before the output they redirected.
  // `correction_offsets` (assistant-text length at each accepted correction)
  // carries that boundary; older gateways omit it.
  const rawCorrections = projection.inflight?.corrections ?? []
  const rawOffsets = projection.inflight?.correction_offsets

  const inflightCorrectionEntries = rawCorrections
    .map((correction, index) => ({ text: correction?.trim() ?? '', offset: rawOffsets?.[index] }))
    .filter(entry => entry.text)

  const inflightCorrections = inflightCorrectionEntries.map(entry => entry.text)

  const correctionOffsetsUsable =
    inflightCorrectionEntries.length > 0 &&
    inflightCorrectionEntries.every(entry => typeof entry.offset === 'number' && entry.offset >= 0)

  // A retained failed turn (the gateway keeps error snapshots replayable when
  // the terminal frame may have been lost to a disconnect) — surface the
  // failure on the projected row instead of rendering the partial as healthy.
  const inflightError = projection.inflight?.error?.trim() ?? ''
  const inflightErrorSurface = parseErrorSurface(projection.inflight?.error_surface)
  const queuedUser = projection.queued?.user?.trim() ?? ''

  if (
    !inflightUser &&
    !inflightAssistant &&
    !inflightStreaming &&
    !inflightError &&
    !queuedUser &&
    !inflightCorrections.length
  ) {
    return messages
  }

  const sessionId = projection.session_id || 'session'
  const projected: ChatMessage[] = []
  // A turn normally persists its user row before inference begins. session.resume
  // then returns that stored row *and* the still-live inflight projection; adding
  // both makes a backgrounded prompt appear twice when its session is reopened.
  // Only suppress the projection when the latest authoritative user row is the
  // same turn — older identical prompts must not hide a newly accepted repeat.
  // A mid-turn redirect gives that turn a RUN of user rows (prompt +
  // corrections). Arrival order seals already-streamed output BETWEEN those
  // rows (#73793), so collect the run by walking back over the live tail:
  // user rows count, live-tail assistant rows are skipped, and a committed
  // assistant reply ends the turn.
  const latestUserIndex = messages.map(message => message.role).lastIndexOf('user')
  const latestUserRun: ChatMessage[] = []

  for (let index = latestUserIndex; index >= 0; index -= 1) {
    const candidate = messages[index]

    if (candidate.role === 'user') {
      latestUserRun.unshift(candidate)

      continue
    }

    if (candidate.role === 'assistant' && isLiveTailRow(candidate)) {
      continue
    }

    break
  }

  const persistedInLatestRun = (text: string): boolean =>
    latestUserRun.some(
      message => textWithoutReferenceLines(chatMessageText(message)) === textWithoutReferenceLines(text)
    )

  const inflightUserAlreadyPersisted =
    projection[safelyPersistedInflightUser] === true || (Boolean(inflightUser) && persistedInLatestRun(inflightUser))

  if (inflightUser && !inflightUserAlreadyPersisted) {
    // Project the prompt through the same conversion history uses, so the live
    // bubble matches its persisted twin: attachment refs lift into the chip row,
    // and a synthetic starting prompt (process_complete, hidden, …) takes the
    // display typing its row will get (#112144) — `hidden` yields nothing.
    const displayKind = projection.inflight?.display_kind

    const typed = toChatMessages([
      {
        role: 'user',
        content: inflightUser,
        ...(displayKind ? { display_kind: displayKind } : {}),
        ...(displayKind && projection.inflight?.display_metadata !== undefined
          ? { display_metadata: projection.inflight.display_metadata }
          : {})
      }
    ])

    projected.push(...typed.map(message => ({ ...message, id: `user-inflight-${sessionId}` })))
  }

  // Keep a pending assistant boundary even before the first delta when a
  // queued user turn follows it. This preserves the two distinct turns.
  //
  // When the *current live turn* already holds a structured mid-turn assistant
  // row (reasoning / tool-call from the live stream or journal), do NOT append
  // a pure-text projection of `inflight.assistant` — that flat dump re-renders
  // thinking as answer text and sandwiches the structured parts (#76444).
  // Only inspect the live tail after the latest user run — never a completed
  // historical tool-bearing reply earlier in the transcript (review feedback).
  const liveStreamId = `assistant-stream-${sessionId}`

  const liveAssistantOfCurrentTurn = ((): ChatMessage | null => {
    const byStreamId = messages.find(message => message.id === liveStreamId)

    if (byStreamId) {
      return byStreamId
    }

    // Assistants after the latest user row belong to this turn's tail.
    if (latestUserIndex < 0) {
      return null
    }

    for (let index = messages.length - 1; index > latestUserIndex; index -= 1) {
      if (messages[index].role === 'assistant') {
        return messages[index]
      }
    }

    return null
  })()

  const turnAlreadyStructured = Boolean(
    liveAssistantOfCurrentTurn &&
    hasStructuralParts(liveAssistantOfCurrentTurn) &&
    isLiveTailRow(liveAssistantOfCurrentTurn)
  )

  // An activate snapshot taken before the turn committed goes stale once REST
  // returns the committed reply after the persisted prompt: its partial
  // `inflight.assistant` is a prefix of that reply, and projecting it paints
  // the answer twice (the extra row frozen on its first chunk). Text alone
  // cannot tell this turn's reply from the previous turn's answer to the same
  // resent prompt, so the reply must have been written after this turn began.
  const turnStartedAt = projection.turn_started_at
  const committedAt = liveAssistantOfCurrentTurn?.timestamp

  const turnAlreadyCommitted = Boolean(
    inflightUserAlreadyPersisted &&
    !inflightError &&
    liveAssistantOfCurrentTurn &&
    !isLiveTailRow(liveAssistantOfCurrentTurn) &&
    committedDuringTurn(turnStartedAt, committedAt) &&
    committedReplyCovers(liveAssistantOfCurrentTurn, inflightAssistant)
  )

  const wantsAssistantRow = Boolean(
    inflightAssistant || inflightStreaming || inflightError || (inflightUser && queuedUser)
  )

  const projectAssistantDump = wantsAssistantRow && !turnAlreadyCommitted && !(turnAlreadyStructured && !inflightError)

  // #121122, the mirror of turnAlreadyCommitted: REST already holds this
  // turn's PARTIAL assistant row (committed as the turn progressed, tool
  // blocks included) while `inflight` still streams the fuller dump.
  // Appending the dump paints the turn twice — the frozen partial with its
  // action bar plus the live copy repeating it. Fold the dump into the tail
  // row instead: same live id (deltas keep landing), fuller text, committed
  // structure carried over. Only when the dump extends the tail text — a
  // diverged tail is a different reply and both rows survive.
  const committedPartial =
    liveAssistantOfCurrentTurn && !isLiveTailRow(liveAssistantOfCurrentTurn) ? liveAssistantOfCurrentTurn : null

  const committedPartialAt = committedPartial ? messages.lastIndexOf(committedPartial) : -1

  const committedPartialText = committedPartial ? chatMessageText(committedPartial) : ''

  const turnPartiallyCommitted = Boolean(
    projectAssistantDump &&
    !inflightError &&
    inflightStreaming &&
    committedPartial &&
    inflightUserAlreadyPersisted &&
    !correctionOffsetsUsable &&
    committedDuringTurn(turnStartedAt, committedAt) &&
    (committedPartialText.trim() === inflightAssistant.trim() ||
      isStrictAnswerTextExtension(inflightAssistant, committedPartialText))
  )

  const foldTarget = turnPartiallyCommitted ? committedPartial : null

  const pushCorrection = (correction: string, index: number): void => {
    if (persistedInLatestRun(correction)) {
      return
    }

    projected.push({
      id: `user-inflight-correction-${index}-${sessionId}`,
      role: 'user',
      parts: [textPart(correction)]
    })
  }

  // Corrections typed while the turn ran are ordered by ARRIVAL: each lands
  // after the assistant output that had already streamed when it was typed and
  // before the output it redirected (#73793 — the old prompt → corrections →
  // reply order spliced them above screens of output the user had already
  // read). With usable offsets the flat dump is split at each boundary; without
  // them (older gateway, or a structured/error tail that must stay whole) the
  // corrections follow the projected reply, matching the live transcript's
  // append-at-tail contract.
  if (projectAssistantDump && correctionOffsetsUsable && !inflightError && inflightAssistant) {
    let cursor = 0

    for (const [index, entry] of inflightCorrectionEntries.entries()) {
      const boundary = Math.min(Math.max(entry.offset as number, cursor), inflightAssistant.length)
      const segment = inflightAssistant.slice(cursor, boundary)

      if (segment.trim()) {
        // Sealed pre-correction output. The `inflight-assistant-` prefix marks
        // it a live-tail row so repeated resumes keep the user run intact.
        projected.push({
          id: `inflight-assistant-segment-${index}-${sessionId}`,
          role: 'assistant',
          parts: [assistantTextPart(segment)],
          pending: false,
          interim: true
        })
      }

      cursor = boundary
      pushCorrection(entry.text, index)
    }

    const tail = inflightAssistant.slice(cursor)

    projected.push({
      id: liveStreamId,
      role: 'assistant',
      parts: tail.trim() ? [assistantTextPart(tail)] : [],
      pending: inflightStreaming
    })
  } else {
    if (projectAssistantDump) {
      const liveRow: ChatMessage = {
        id: liveStreamId,
        role: 'assistant',
        parts: inflightAssistant ? [assistantTextPart(inflightAssistant)] : [],
        pending: inflightStreaming,
        ...(inflightError ? { error: inflightError } : {}),
        ...(inflightError && inflightErrorSurface ? { errorSurface: inflightErrorSurface } : {})
      }

      if (foldTarget) {
        // #121122: the persisted tail IS this turn's partial — replace it in
        // place so the transcript holds one row that keeps streaming.
        // Structure the flat dump cannot express (tool calls, reasoning)
        // carries over; row id and reactions stay so nothing blinks off
        // mid-turn and a reaction toggle still reaches the persisted row.
        projected.push(carryRowIdentity(preserveStructuralParts(liveRow, foldTarget), foldTarget))
      } else {
        projected.push(liveRow)
      }
    }

    for (const [index, correction] of inflightCorrections.entries()) {
      pushCorrection(correction, index)
    }
  }

  if (queuedUser) {
    projected.push({
      id: `user-queued-${sessionId}`,
      role: 'user',
      parts: [textPart(queuedUser)]
    })
  }

  if (foldTarget) {
    // Splice the folded live row (plus any corrections/queued tail) into the
    // committed partial's slot instead of appending beside it.
    return [...messages.slice(0, committedPartialAt), ...projected, ...messages.slice(committedPartialAt + 1)]
  }

  return projected.length ? [...messages, ...projected] : messages
}

function normalizedMessageText(message: ChatMessage): string {
  return chatMessageText(message).replace(/\s+/g, ' ').trim()
}

function transcriptAnchorMatches(a: ChatMessage, b: ChatMessage): boolean {
  if (a.role !== b.role) {
    return false
  }

  const aText = normalizedMessageText(a)
  const bText = normalizedMessageText(b)

  if (a.timestamp !== undefined && b.timestamp !== undefined) {
    return a.timestamp === b.timestamp && aText === bText
  }

  return Boolean(aText) && aText === bText
}

/**
 * Mark only an already-materialized `inflight.user` for visual suppression.
 *
 * A running gateway returns two independent truths: its compressed runtime
 * history plus the current in-flight turn, while REST may already have flushed
 * that user row into the complete persisted transcript. Global text dedupe is
 * unsafe because users may intentionally submit the same prompt twice. Instead,
 * find the last runtime message inside the persisted transcript and inspect only
 * the newer persisted suffix.
 *
 * Keep `inflight.user` intact because it also carries turn structure: a queued
 * prompt needs its assistant boundary even when the persisted user has no
 * assistant delta yet. The private marker lets the renderer suppress only that
 * duplicate bubble. If the histories have no safe common anchor, keep the
 * projection unchanged — a duplicate is recoverable, but dropping a real
 * accepted prompt is not.
 */
export function dedupeInflightUserAgainstTranscript(
  persistedMessages: ChatMessage[],
  runtimeMessages: ChatMessage[],
  projection: SessionResumeResult
): ReconciledSessionResumeResult {
  const inflightUser = projection.inflight?.user?.replace(/\s+/g, ' ').trim() ?? ''

  if (!inflightUser) {
    return projection
  }

  let suffixStart = 0

  if (runtimeMessages.length) {
    const runtimeAnchor = runtimeMessages[runtimeMessages.length - 1]
    let persistedAnchorIndex = -1

    for (let index = persistedMessages.length - 1; index >= 0; index -= 1) {
      if (transcriptAnchorMatches(persistedMessages[index], runtimeAnchor)) {
        persistedAnchorIndex = index

        break
      }
    }

    if (persistedAnchorIndex < 0) {
      return projection
    }

    suffixStart = persistedAnchorIndex + 1
  }

  const persistedTail = persistedMessages.slice(suffixStart)
  const lastPersistedMessage = persistedTail[persistedTail.length - 1]

  const persistedUserPresent =
    lastPersistedMessage?.role === 'user' && normalizedMessageText(lastPersistedMessage) === inflightUser

  if (!persistedUserPresent) {
    return projection
  }

  return { ...projection, [safelyPersistedInflightUser]: true }
}

/**
 * Drop only synthetic local tail rows that the activation snapshot replaces.
 * Unmatched optimistic rows survive so a submit racing with activation is not
 * lost; completed transcript rows before the open tail are never considered.
 */
export function removeRepresentedLocalLiveProjection(
  previousMessages: ChatMessage[],
  projection: Pick<SessionResumeResult, 'inflight' | 'queued'>
): ChatMessage[] {
  const inflightUser = projection.inflight?.user?.replace(/\s+/g, ' ').trim() ?? ''
  const inflightAssistant = projection.inflight?.assistant?.replace(/\s+/g, ' ').trim() ?? ''
  const queuedUser = projection.queued?.user?.replace(/\s+/g, ' ').trim() ?? ''

  const hasAssistantProjection = Boolean(
    projection.inflight?.assistant || projection.inflight?.streaming || (inflightUser && queuedUser)
  )

  if (!inflightUser || !hasAssistantProjection) {
    return previousMessages
  }

  let openTailStart = 0

  for (let index = previousMessages.length - 1; index >= 0; index -= 1) {
    const message = previousMessages[index]

    if (message.role === 'assistant' && !message.pending) {
      openTailStart = index + 1

      break
    }
  }

  const inflightUserIndex = previousMessages.findIndex(
    (message, index) =>
      index >= openTailStart &&
      message.role === 'user' &&
      message.id.startsWith('user-') &&
      normalizedMessageText(message) === inflightUser
  )

  const assistantIndex = inflightUserIndex + 1
  const assistant = previousMessages[assistantIndex]
  const localAssistant = assistant ? normalizedMessageText(assistant) : ''

  // The activation snapshot and this local row are read at different times
  // while the turn keeps streaming in the background, so neither is
  // guaranteed to be textually identical to the other even though both
  // represent the same running reply — one is simply further along.
  const assistantTextRepresented =
    localAssistant === inflightAssistant ||
    isStrictAnswerTextExtension(localAssistant, inflightAssistant) ||
    isStrictAnswerTextExtension(inflightAssistant, localAssistant)

  const assistantMatches =
    inflightUserIndex >= openTailStart &&
    assistant?.role === 'assistant' &&
    assistant.id.startsWith('assistant-stream-') &&
    assistantTextRepresented

  if (!assistantMatches) {
    return previousMessages
  }

  let queuedUserIndex = -1

  if (queuedUser) {
    queuedUserIndex = previousMessages.findIndex(
      (message, index) =>
        index > assistantIndex &&
        message.role === 'user' &&
        message.id.startsWith('user-queued-') &&
        normalizedMessageText(message) === queuedUser
    )
  }

  return previousMessages.filter(
    (_message, index) => index !== inflightUserIndex && index !== assistantIndex && index !== queuedUserIndex
  )
}

/**
 * Overlay messages that changed while activation waited on REST. Existing ids
 * replace the older activation row; only rows added or changed since the warm
 * cache baseline are appended. This is identity-based, never text-based.
 */
export function overlayConcurrentMessageChanges(
  nextMessages: ChatMessage[],
  baselineMessages: ChatMessage[],
  currentMessages: ChatMessage[]
): ChatMessage[] {
  const baselineById = new Map(baselineMessages.map(message => [message.id, message]))
  const nextIndexById = new Map(nextMessages.map((message, index) => [message.id, index]))
  let changed = false
  const overlaid = [...nextMessages]
  const dropped = new Set<string>()

  let activationStreamIndex = overlaid.findIndex(
    message =>
      message.role === 'assistant' && message.id.startsWith('assistant-stream-') && !baselineById.has(message.id)
  )

  // message.complete settled this stream row while REST was in flight, and
  // the page already carries the same reply under its committed id (#70209).
  // Only a row the page newly added counts: one already in the baseline is an
  // earlier turn's answer (a resent prompt can repeat it word for word). An
  // errored row carries a failure the committed text cannot show.
  const committedOnPage = (current: ChatMessage): boolean => {
    const text = textWithoutReferenceLines(chatMessageText(current)).trim()
    const lastUser = overlaid.findLastIndex(message => message.role === 'user')
    const liveRows = transcriptRowIds(current)

    return (
      Boolean(text) &&
      overlaid.some((message, index) => {
        if (
          !(index > lastUser) ||
          message.role !== 'assistant' ||
          baselineById.has(message.id) ||
          isLiveTailRow(message)
        ) {
          return false
        }

        // The completion bound the reply to its stored row. A tool turn's
        // history folds that row into the bubble that opened the turn, whose
        // text also carries the narration, so match the row, not the words.
        if (liveRows.length && transcriptRowIds(message).some(id => liveRows.includes(id))) {
          return true
        }

        // The committed row and the settled live row capture the same reply
        // at two moments while it kept streaming, so neither side is
        // guaranteed to be textually identical: accept either as a forward
        // text-extension of the other, the same trade
        // removeRepresentedLocalLiveProjection made in 2494b95929. A folded
        // tool turn is compared by its answer, the text after its last tool.
        return [chatMessageText(message), replyAfterLastTool(message)].some(full => {
          const candidate = textWithoutReferenceLines(full).trim()

          return (
            candidate === text ||
            isStrictAnswerTextExtension(candidate, text) ||
            isStrictAnswerTextExtension(text, candidate)
          )
        })
      })
    )
  }

  const representedByStoredRow = (current: ChatMessage): boolean => {
    const rows = transcriptRowIds(current)

    return (
      rows.length > 0 &&
      rows.every(id =>
        overlaid.some(
          message =>
            message.id !== current.id && message.role === current.role && transcriptRowIds(message).includes(id)
        )
      )
    )
  }

  const representedOnPage = (current: ChatMessage): boolean =>
    // The page can still carry the row's older streaming copy by id when it
    // was composed from the baseline; the committed row replaces both.
    (current.role === 'assistant' &&
      current.pending !== true &&
      !current.error &&
      isLiveTailReplyId(current.id) &&
      committedOnPage(current)) ||
    // A completion receipt bound this optimistic prompt to its stored row
    // while REST was in flight, so it differs from the pre-bind baseline only
    // by that row id. The page already carries the stored row under its own
    // id: the prompt is represented, not a new concurrent row.
    (current.role === 'user' && representedByStoredRow(current))

  for (const current of currentMessages) {
    const baseline = baselineById.get(current.id)
    const changedSinceBaseline = !baseline || !chatMessagesEquivalent(baseline, current)

    if (!changedSinceBaseline) {
      continue
    }

    const nextIndex = nextIndexById.get(current.id)

    if (representedOnPage(current)) {
      if (nextIndex !== undefined) {
        dropped.add(current.id)
      }

      continue
    }

    if (nextIndex !== undefined) {
      if (!chatMessagesEquivalent(overlaid[nextIndex], current)) {
        overlaid[nextIndex] = current
        changed = true
      }

      continue
    }

    if (activationStreamIndex >= 0 && current.role === 'assistant' && current.id.startsWith('assistant-stream-')) {
      const activationStream = overlaid[activationStreamIndex]
      const activationText = chatMessageText(activationStream)
      const currentText = chatMessageText(current)

      const replacement =
        activationText && !currentText.startsWith(activationText)
          ? { ...current, parts: [...activationStream.parts, ...current.parts] }
          : current

      nextIndexById.delete(activationStream.id)
      nextIndexById.set(current.id, activationStreamIndex)
      overlaid[activationStreamIndex] = replacement
      activationStreamIndex = -1
      changed = true

      continue
    }

    nextIndexById.set(current.id, overlaid.length)
    overlaid.push(current)
    changed = true
  }

  // A live row the store retired while REST was in flight (message.complete
  // settled it into the committed reply) stays retired: the page was composed
  // from the older baseline, so it can still carry that row by id.
  const currentIds = new Set(currentMessages.map(message => message.id))

  const retired = (message: ChatMessage) =>
    dropped.has(message.id) || (isLiveTailRow(message) && baselineById.has(message.id) && !currentIds.has(message.id))

  if (currentMessages.length && overlaid.some(retired)) {
    return overlaid.filter(message => !retired(message))
  }

  return changed ? overlaid : nextMessages
}

export interface BranchMessage {
  content: string
  role: ChatMessage['role']
  source: ChatMessage
}

// The copyable spine of a branch: user/assistant turns that carry text.
export const toBranchMessages = (messages: ChatMessage[]): BranchMessage[] =>
  messages
    .map(message => ({ content: chatMessageText(message), role: message.role, source: message }))
    .filter(({ content, role }) => content.trim() && (role === 'assistant' || role === 'user'))

/**
 * Choose the transcript used to seed an open-chat branch.
 *
 * The local renderer can hold a compacted model projection, while the REST
 * transcript contains the complete display projection. Use the latter for a
 * whole-chat branch. When branching from a clicked bubble, map that bubble by
 * durable row id first and by same-role/text ordinal as a legacy fallback; if
 * it cannot be mapped, keep the local prefix rather than silently choosing a
 * different point in the conversation.
 */
export function selectBranchMessages(
  localMessages: ChatMessage[],
  authoritativeMessages: ChatMessage[] | null,
  messageId?: string
): BranchMessage[] {
  const localIndex = messageId ? localMessages.findIndex(message => message.id === messageId) : -1

  if (!authoritativeMessages?.length) {
    return toBranchMessages(localMessages.slice(0, localIndex >= 0 ? localIndex + 1 : localMessages.length))
  }

  if (!messageId) {
    return toBranchMessages(authoritativeMessages)
  }

  if (localIndex < 0) {
    return toBranchMessages(localMessages)
  }

  const target = localMessages[localIndex]

  let authoritativeIndex =
    target.rowId === undefined
      ? -1
      : authoritativeMessages.findIndex(message => message.rowId !== undefined && message.rowId === target.rowId)

  // Strip `@image:` directive lines the same way the persisted→ChatMessage
  // conversion does (extractImageRefs lifts them into attachmentRefs), so a
  // local optimistic bubble and its authoritative twin compare equal.
  const comparableText = (message: ChatMessage) =>
    textWithoutEmbeddedImages(chatMessageText(message))
      .replace(/^@image:[^\n]*\n?/gm, '')
      .trim()

  if (authoritativeIndex < 0) {
    const targetText = comparableText(target)

    const targetOrdinal = localMessages
      .slice(0, localIndex + 1)
      .filter(message => message.role === target.role && comparableText(message) === targetText).length

    let ordinal = 0

    authoritativeIndex = authoritativeMessages.findIndex(message => {
      if (message.role !== target.role || comparableText(message) !== targetText) {
        return false
      }

      ordinal += 1

      return ordinal === targetOrdinal
    })
  }

  if (authoritativeIndex < 0) {
    return toBranchMessages(localMessages.slice(0, localIndex + 1))
  }

  return toBranchMessages(authoritativeMessages.slice(0, authoritativeIndex + 1))
}

export function upsertOptimisticSession(
  created: SessionCreateResponse,
  id: string,
  title: string | null = null,
  preview: string | null = null,
  parentSessionId: string | null = null,
  lastActive?: number,
  owner?: null | SessionProfileRoute
) {
  const session = buildOptimisticSession(created, id, title, preview, parentSessionId, lastActive, owner)

  if (owner) {
    setSessionOwnerHint(id, owner)
  }

  // A real row supersedes any unlisted-draft stub for the same id (first send
  // on a ⌘T tab lists it); drop the stub so the atom stays bounded. The
  // lookup shadow-filter covers any path that lists without passing here.
  setUnlistedSessionOwnerRows(prev => (prev.some(s => s.id === id) ? prev.filter(s => s.id !== id) : prev))

  setSessions(prev => [session, ...prev.filter(s => s.id !== id)])
}

/**
 * Record the owner of an UNLISTED draft tile without touching the visible
 * sidebar list (see `openNewSessionTile` with `listed: false`, #102792). The
 * stub carries the same stamps an optimistic row would — ambient profile for
 * an unrouted create, exact connection tag for a routed one — and rides only
 * the owner-lookup path, never the sidebar render path. A later real row for
 * the same id shadows it; the first send replaces it outright.
 */
export function upsertUnlistedSessionOwner(
  created: SessionCreateResponse,
  id: string,
  owner?: null | SessionProfileRoute
) {
  const stub = buildOptimisticSession(created, id, null, null, null, undefined, owner)
  setSessionOwnerHintForStub(id, owner)
  setUnlistedSessionOwnerRows(prev => [stub, ...prev.filter(s => s.id !== id)])
}

function setSessionOwnerHintForStub(id: string, owner?: null | SessionProfileRoute): void {
  if (owner) {
    setSessionOwnerHint(id, owner)
  }
}

function buildOptimisticSession(
  created: SessionCreateResponse,
  id: string,
  title: string | null = null,
  preview: string | null = null,
  parentSessionId: string | null = null,
  lastActive?: number,
  owner?: null | SessionProfileRoute
): SessionInfo {
  const now = lastActive ?? Date.now() / 1000
  // Stamp the profile the session was just created on so the scoped sidebar
  // shows the new row immediately instead of filtering it out as "default"
  // until the aggregator re-fetches. An explicitly routed create ($newChatRoute
  // / a tile's route) names its EXACT owner: the backend profile that route
  // serves, on that route's connection. The live gateway's profile is only the
  // owner for an unrouted create — in All-profiles / Bot routing the ambient
  // profile stays on `default` while the session lives on another backend (and
  // a concurrent source switch can move the active gateway before this row is
  // inserted), so a row stamped `default` then misroutes every session-scoped
  // RPC that resolves its owner off the row ("session not found" on turn two).
  const profileKey = normalizeProfileKey(owner ? owner.targetProfile || owner.profile : $activeGatewayProfile.get())
  const connectionId = owner?.connectionId.trim() || ''

  const session: SessionInfo = {
    // Seed cwd so the grouped sidebar can place the new row in its repo/worktree
    // lane immediately (the overlay groups by path); fall back to the workspace
    // the session was just started in when the create response omits it.
    cwd: created.info?.cwd ?? ($currentCwd.get().trim() || null),
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: true,
    is_default_profile: profileKey === 'default',
    last_active: now,
    message_count: created.message_count ?? created.messages?.length ?? 0,
    model: created.info?.model ?? null,
    output_tokens: 0,
    parent_session_id: parentSessionId,
    ...(parentSessionId ? { _branched_from: parentSessionId } : {}),
    preview,
    profile: profileKey,
    source: 'tui',
    started_at: now,
    title,
    tool_call_count: 0,
    ...(connectionId ? { connection_id: connectionId } : {})
  }

  return session
}

export function patchSessionWorkspace(sessionId: string, cwd: string | undefined) {
  if (!cwd) {
    return
  }

  setSessions(prev => prev.map(session => (session.id === sessionId ? { ...session, cwd } : session)))
}

export function sessionShouldHaveTranscript(session: SessionInfo | undefined): boolean {
  return (session?.message_count ?? 0) > 0
}

export type ListedSessionSlice = 'cron' | 'messaging' | 'sessions'

export function findListedSession(
  storedSessionId: string
): { session: SessionInfo; slice: ListedSessionSlice } | undefined {
  const match = (session: SessionInfo) => sessionMatchesStoredId(session, storedSessionId)
  const fromMessaging = $messagingSessions.get().find(match)

  if (fromMessaging) {
    return { session: fromMessaging, slice: 'messaging' }
  }

  const fromCron = $cronSessions.get().find(match)

  if (fromCron) {
    return { session: fromCron, slice: 'cron' }
  }

  const fromSessions = $sessions.get().find(match)

  if (fromSessions) {
    return { session: fromSessions, slice: 'sessions' }
  }

  return undefined
}

export function dropListedSession(storedSessionId: string): void {
  const keep = (session: SessionInfo) => !sessionMatchesStoredId(session, storedSessionId)

  setSessions(prev => prev.filter(keep))
  setMessagingSessions(prev => prev.filter(keep))
  setCronSessions(prev => prev.filter(keep))
  setUnlistedSessionOwnerRows(prev => prev.filter(keep))
}

export function listedSliceTarget(session: SessionInfo): ListedSessionSlice {
  return isMessagingSource(session.source)
    ? 'messaging'
    : normalizeSessionSource(session.source) === 'cron'
      ? 'cron'
      : 'sessions'
}

export function restoreListedSession(session: SessionInfo, slice?: ListedSessionSlice): void {
  const target: ListedSessionSlice = slice ?? listedSliceTarget(session)

  const prepend = (prev: SessionInfo[]) => [
    session,
    ...prev.filter(existing => !sessionMatchesStoredId(existing, session.id))
  ]

  if (target === 'messaging') {
    setMessagingSessions(prepend)

    return
  }

  if (target === 'cron') {
    setCronSessions(prepend)

    return
  }

  setSessions(prepend)
}

function upsertResolvedSession(
  session: SessionInfo,
  storedSessionId: string,
  tombstoneGenerationsAtRequestStart: SessionTombstoneGenerationSnapshot
) {
  // Exact-id lookup intentionally resolves internal delegate children so a
  // watch tile can open them. They are not ordinary user conversations,
  // though, and the authoritative list endpoints omit them; caching one here
  // would bypass that boundary and leak it into the Sessions sidebar.
  if (session.is_internal_child) {
    return
  }

  const removed = $removedSessionIds.get()
  const identities = [storedSessionId, session.id, session._lineage_root_id]

  // A direct by-id resolve may have started just before an archive/delete
  // (#85163: the archive row click's bubbled resume raced the tombstone).
  // A stale response must not undo the optimistic eviction while the mutation's
  // tombstone is active, after the tombstone was already present at request
  // start, or after an add → remove ABA cycle made membership look unchanged.
  // Check every identity lineage-aware lookups use. This suppresses only the
  // sidebar-cache upsert: the resolved row is still returned so an explicit
  // resume-by-id can open archived history, and a later request after a
  // settled rollback can publish normally.
  if (
    session.archived ||
    identities.some(id => (id ? removed.has(id) : false)) ||
    tombstoneLifecycleChanged(tombstoneGenerationsAtRequestStart, identities)
  ) {
    return
  }

  const lineage = session._lineage_root_id ?? session.id

  // A hidden row (canonical Bot Chat, room plumbing) is unlisted by design:
  // inserting it into $sessions paints a sidebar row until the next refresh,
  // and the keep-list then holds it there (#113273). Park it on the off-list
  // owner atom the draft stubs ride — owner resolution still finds it via
  // ownerLookupSessionRows, the sidebar never does.
  if (session.hidden) {
    setUnlistedSessionOwnerRows(prev => [
      session,
      ...prev.filter(existing => (existing._lineage_root_id ?? existing.id) !== lineage)
    ])

    return
  }

  const prepend = (prev: SessionInfo[]) => [
    session,
    ...prev.filter(existing => {
      if (sessionMatchesStoredId(existing, storedSessionId)) {
        return false
      }

      return (existing._lineage_root_id ?? existing.id) !== lineage
    })
  ]

  // A resolve can observe a source move (cross-room /resume rewrites the row to
  // source='matrix', #113827): the row belongs to its current slice, and the
  // stale copy in every other slice must go or the session shows twice.
  // Identity-stable when nothing matched — every sidebar memo keys on these
  // arrays, and a resolve runs on each row open.
  const evict = (prev: SessionInfo[]) =>
    prev.some(existing => sessionMatchesStoredId(existing, storedSessionId))
      ? prev.filter(existing => !sessionMatchesStoredId(existing, storedSessionId))
      : prev

  const target = listedSliceTarget(session)

  setSessions(target === 'sessions' ? prepend : evict)
  setMessagingSessions(target === 'messaging' ? prepend : evict)
  setCronSessions(target === 'cron' ? prepend : evict)
}

// Every session row reachable through the profile-scoped project tree —
// preview rows on a collapsed project plus the drill-in lane rows. These are
// the only rows guaranteed to name their owning profile (the gateway stamps
// the request scope onto them), so owner resolution has to see them.
function projectTreeSessions(): SessionInfo[] {
  return $projectTree
    .get()
    .flatMap(project => [
      ...(project.previewSessions ?? []),
      ...project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))
    ])
}

// The best cached row for a stored id, across every list that can hold one.
// "Best" means self-describing: the same conversation can appear both as an
// ownerless legacy Recents copy and as a profile-stamped project-tree row, and
// picking the ownerless one throws away the only routing information we have.
export function cachedSessionRow(storedSessionId: string): SessionInfo | undefined {
  const candidates = [
    ...$sessions.get(),
    ...$cronSessions.get(),
    ...$messagingSessions.get(),
    ...projectTreeSessions()
  ].filter(session => sessionMatchesStoredId(session, storedSessionId))

  return (
    candidates.find(session => session.connection_id?.trim()) ??
    candidates.find(session => session.profile?.trim()) ??
    candidates[0]
  )
}

export type StoredSessionProbe = { status: 'found'; session: SessionInfo } | { status: 'gone' | 'inconclusive' }

export async function resolveStoredSession(
  storedSessionId: string,
  ownerRoute?: SessionProfileRoute
): Promise<SessionInfo | undefined> {
  const result = await probeStoredSession(storedSessionId, ownerRoute)

  return result.status === 'found' ? result.session : undefined
}

/** `resolveStoredSession` with the ladder's evidence kept: `gone` only when
 *  every rung answered an explicit session 404 (a 5xx, a network failure or a
 *  bare 404 from a proxy is `inconclusive`), and — without an owner — only once
 *  the profile inventory is known, or a single-profile sweep would vouch for
 *  ids that live on a profile not yet listed (#125678). */
export async function probeStoredSession(
  storedSessionId: string,
  ownerRoute?: SessionProfileRoute
): Promise<StoredSessionProbe> {
  let allGone = true

  const recordFailure = (error: unknown) => {
    const message = error instanceof Error ? error.message : String(error ?? '')
    allGone &&= /\b404\b/.test(message) && /session not found/i.test(message)
  }

  // Snapshot BEFORE any await: a resolve that started before an archive/delete
  // must reject its own stale response (see upsertResolvedSession).
  const tombstoneGenerationsAtRequestStart = captureSessionTombstoneGenerations()

  const cached = cachedSessionRow(storedSessionId)

  if (ownerRoute) {
    const scope = {
      connectionId: ownerRoute.connectionId,
      profile: ownerRoute.targetProfile || ownerRoute.profile
    }

    const cachedOwnerMatches =
      cached &&
      cached.connection_id === ownerRoute.connectionId &&
      (!cached.profile || normalizeProfileKey(cached.profile) === normalizeProfileKey(ownerRoute.profile))

    if (cached && cachedOwnerMatches) {
      return { status: 'found', session: cached }
    }

    try {
      const session = await getSession(storedSessionId, scope)
      session.profile = normalizeProfileKey(ownerRoute.profile)
      session.connection_id = ownerRoute.connectionId
      upsertResolvedSession(session, storedSessionId, tombstoneGenerationsAtRequestStart)

      return { status: 'found', session }
    } catch (error) {
      // An explicit owner is fail-closed. Probing the ambient or another
      // profile would turn a stale route into a cross-connection open.
      recordFailure(error)

      return { status: allGone ? 'gone' : 'inconclusive' }
    }
  }

  // A row with no owning profile can't route a resume when more than one
  // profile exists — a resume without a profile lands on whichever gateway is
  // active (#67603 family, cross-profile open asymmetry). Treat such a hit as
  // unresolved and fall through to the by-id lookups, which stamp ownership.
  const multiProfile = $profiles.get().length > 1

  if (cached && (cached.profile?.trim() || !multiProfile)) {
    return { status: 'found', session: cached }
  }

  // Direct by-id on the active profile — one row lookup, no list scan. Electron
  // routes an unscoped GET to the primary backend, which may not own the
  // active profile. A 404 there used to skip that profile in the probes below,
  // so the session was never found.
  const activeKey = normalizeProfileKey($activeGatewayProfile.get())

  try {
    const session = await getSession(storedSessionId, activeKey)

    // Older backends can omit `profile`; this request targeted the active
    // profile, so back-fill that rather than caching an unowned row. A present
    // stamp is preserved for backend compatibility.
    session.profile ||= activeKey

    upsertResolvedSession(session, storedSessionId, tombstoneGenerationsAtRequestStart)

    return { status: 'found', session }
  } catch (error) {
    // Not on the active profile — fall through to the cross-profile probe.
    recordFailure(error)
  }

  // Multi-profile only: probe each remaining profile by id (still one cheap
  // lookup each) rather than pulling every profile's recent sessions. The
  // first hit carries its owning `profile`, which routes the resume to the
  // right backend. The active profile was already tried above.
  const otherProfiles = $profiles
    .get()
    .map(profile => normalizeProfileKey(profile.name))
    .filter(key => key !== activeKey)

  for (const profile of otherProfiles) {
    try {
      const session = await getSession(storedSessionId, profile)

      // Same ownership contract: the DESKTOP profile we explicitly probed is
      // authoritative, whatever the scoped backend stamped (older backends
      // omit the field; a per-profile remote override strips the alias before
      // forwarding, so that backend answers as its own "default").
      session.profile = profile

      upsertResolvedSession(session, storedSessionId, tombstoneGenerationsAtRequestStart)

      return { status: 'found', session }
    } catch (error) {
      // Not on this profile; try the next.
      recordFailure(error)
    }
  }

  return { status: allGone && $profiles.get().length > 0 ? 'gone' : 'inconclusive' }
}

/**
 * The profile that owns a stored session, resolved through the same
 * cache → active-backend → cross-profile ladder as `resolveStoredSession`.
 *
 * Recovery `session.resume` calls (stale runtime id, session-not-found, wedged
 * loop) must re-register the conversation on ITS backend, not on whichever
 * profile happens to be live. Omitting the profile lets the gateway fall back to
 * the launch-profile DB (tui_gateway/server.py), which is how a session bleeds
 * from one profile into another (#67603, second symptom). A cache-only lookup
 * misses any session outside the paginated sidebar window, so route through the
 * resolver, which probes uncached ids across profiles.
 */
export async function resolveSessionProfile(storedSessionId: null | string): Promise<string | undefined> {
  if (!storedSessionId) {
    return undefined
  }

  const profile = (await resolveStoredSession(storedSessionId))?.profile?.trim()

  return profile || undefined
}

/**
 * The OWNER of a stored session through the same cache → active-backend →
 * cross-profile ladder, preferring the EXACT route when the resolved row is
 * connection-tagged (unified-list splice, optimistic create row, a carried
 * tag) over its bare profile. Session-scoped RPC dispatch uses this as the
 * async rung after the sync ladder (tile route → hint → row) misses, so a
 * registry-owned session never degrades to a profile-only route that dials a
 * different socket than the one holding its runtime.
 */
export async function resolveSessionOwner(storedSessionId: null | string): Promise<SessionOwnerScope> {
  if (!storedSessionId) {
    return undefined
  }

  const owner = resolveSessionRpcOwner({
    routingSessionId: storedSessionId,
    tileOwnerRoute: sessionTileOwnerRoute,
    sessionOwnerHint: getSessionOwnerHint,
    sessionRowOwner: id => knownSessionOwner(ownerLookupSessionRows(), id),
    eventOwner: runtimeSessionOwner
  })

  if (owner) {
    return owner
  }

  const row = await resolveStoredSession(storedSessionId)

  return sessionOwnerRouteFromRow(row) ?? (row?.profile?.trim() || undefined)
}

type SessionRuntimeStatePatch = Partial<
  Pick<
    ClientSessionState,
    | 'branch'
    | 'cwd'
    | 'fast'
    | 'model'
    | 'personality'
    | 'provider'
    | 'reasoningEffort'
    | 'reasoningEffortPending'
    | 'reasoningEffortWire'
    | 'serviceTier'
    | 'usage'
    | 'yolo'
  >
>

interface ApplyRuntimeInfoOptions {
  /**
   * Whether this runtime belongs to the session the MAIN pane is showing.
   * Foreground (the default) mirrors into the composer atoms every main-pane
   * surface reads.
   *
   * A tile or a background branch must pass `false`: it owns a different
   * worktree, and writing its cwd into `$currentCwd` re-pointed the main
   * composer's coding rail (and the persisted workspace cwd) at the tile's
   * repo — the main rail painted a branch from a tree its session was never
   * in. The returned patch still carries every field, so the caller's own
   * per-session state is unaffected.
   */
  foreground?: boolean
}

/** Mirror a session's runtime state into the composer atoms the MAIN pane
 *  renders from. Foreground sessions only — see ApplyRuntimeInfoOptions. */
function publishRuntimeToComposer(state: SessionRuntimeStatePatch): void {
  if (state.model !== undefined) {
    setCurrentModel(state.model)
  }

  if (state.provider !== undefined) {
    setCurrentProvider(state.provider)
  }

  if (state.cwd !== undefined) {
    if (state.cwd) {
      // The runtime named a real folder for the session in the main pane, so
      // that conversation owns the path.
      commitWorkspaceCwdForSelectedSession(state.cwd)
    } else {
      // A detached session: the path on screen is provably still the previous
      // conversation's. Release rather than write `''` — clearing it collapses
      // the workspace/review panes on every switch.
      releaseWorkspaceCwdOwner()
    }
  }

  if (state.branch !== undefined) {
    setCurrentBranch(state.branch)
  }

  if (state.personality !== undefined) {
    setCurrentPersonality(state.personality)
  }

  if (state.reasoningEffort !== undefined) {
    setCurrentReasoningEffort(state.reasoningEffort)
  }

  if (state.reasoningEffortWire !== undefined) {
    setCurrentReasoningEffortWire(state.reasoningEffortWire)
  }

  if (state.serviceTier !== undefined) {
    setCurrentServiceTier(state.serviceTier)
  }

  if (state.fast !== undefined) {
    setCurrentFastMode(state.fast)
  }

  if (state.yolo !== undefined) {
    setYoloActive(state.yolo)
  }
}

/** Session-independent side effects of one runtime info snapshot. */
function reportRuntimeInfoAppState(info: SessionRuntimeInfo): void {
  if (info.stored_session_id) {
    reconcilePendingSubmissions(
      info.stored_session_id,
      info.pending_submissions,
      pendingSnapshotFence(info.replay_epoch, info.last_sequence)
    )
  }

  // App/profile-level reporting is session-independent — a tile's runtime
  // reports backend skew and credential warnings just as usefully.
  reportBackendContract(info.desktop_contract, info.desktop_protocol)

  if (info.approval_mode !== undefined) {
    reconcileApprovalModeForProfile($activeGatewayProfile.get(), info.approval_mode)
  }

  requestDesktopOnboardingForCredentialWarning(info.credential_warning)

  reportInstallMethodWarning(info.install_warning)
}

export function applyRuntimeInfo(
  info: SessionRuntimeInfo | undefined,
  { foreground = true }: ApplyRuntimeInfoOptions = {}
): SessionRuntimeStatePatch | null {
  if (!info) {
    return null
  }

  reportRuntimeInfoAppState(info)

  const sessionState: SessionRuntimeStatePatch = {}

  if (typeof info.model === 'string') {
    sessionState.model = info.model
  }

  if (typeof info.provider === 'string') {
    sessionState.provider = info.provider
  }

  // Empty string is authoritative, not "no opinion": a detached/bare session
  // reports `cwd: ''`, and the truthy-only test left `$currentCwd` — and so the
  // Files pane — pinned to the PREVIOUS project for the rest of the session
  // (#71254). Empty is routed through ownership release below rather than
  // persisted, so the pane hides a path it no longer owns instead of blanking.
  if (typeof info.cwd === 'string') {
    sessionState.cwd = info.cwd
  }

  if (info.branch !== undefined) {
    sessionState.branch = info.branch || ''
  }

  if (typeof info.personality === 'string') {
    sessionState.personality = normalizePersonalityValue(info.personality)
  }

  if (typeof info.reasoning_effort === 'string') {
    sessionState.reasoningEffort = info.reasoning_effort
    sessionState.reasoningEffortPending = false
  }

  if (typeof info.reasoning_effort_wire === 'string') {
    sessionState.reasoningEffortWire = info.reasoning_effort_wire
  }

  if (typeof info.service_tier === 'string') {
    sessionState.serviceTier = info.service_tier
  }

  if (typeof info.fast === 'boolean') {
    sessionState.fast = info.fast
  }

  if (typeof info.yolo === 'boolean') {
    sessionState.yolo = info.yolo
  }

  if (info.usage) {
    // Runtime info is an authoritative snapshot. Keep a complete per-runtime
    // usage object so a secondary tile can render immediately after create/resume;
    // an omitted compression count intentionally clears a stale tile value.
    sessionState.usage = {
      ...info.usage,
      calls: info.usage.calls ?? 0,
      compressions: info.usage.compressions,
      input: info.usage.input ?? 0,
      output: info.usage.output ?? 0,
      total: info.usage.total ?? 0
    }
  }

  if (foreground) {
    publishRuntimeToComposer(sessionState)

    if (info.usage) {
      // session.info/session.resume is an authoritative session snapshot, not a
      // partial live tick. Clear a missing compression count so switching from
      // a counted session to an older/cold runtime cannot leak the old value.
      setCurrentUsage(current => ({ ...current, ...info.usage, compressions: info.usage?.compressions }))
    }
  }

  return sessionState
}

export function applyStoredSessionPreviewRuntimeInfo(
  stored: { cwd?: null | string; model?: null | string } | undefined,
  storedSessionId: null | string
) {
  // Compression count is live runtime state, not part of a durable session row.
  // Drop the previous session's value immediately while the selected runtime
  // resumes; the authoritative usage snapshot will repopulate it when present.
  setCurrentUsage(current => ({ ...current, compressions: undefined }))
  // Transient: this is a PREVIEW painted while `session.resume` is still in
  // flight. If the resume is abandoned (user starts a new chat before it
  // returns, or the row is switched away), nothing repairs the selection
  // afterwards — persisting these values left the composer holding a manual
  // model with an EMPTY provider in localStorage, and every later
  // `session.create` paired that model with the profile provider and failed
  // the coherence gate (#125336).
  setCurrentModelTransient(stored?.model || '')
  setCurrentProviderTransient('')
  setCurrentReasoningEffort('')
  setCurrentServiceTier('')
  setCurrentFastMode(false)
  setYoloActive(false)
  setCurrentPersonality('')

  // Cold resume paints the transcript before `session.resume` returns, so
  // without this the Files pane shows the PREVIOUS project's tree for the whole
  // round-trip (#71254 / #76696). The sidebar row already knows this
  // conversation's workspace — `cwd` is part of the compact row projection — so
  // mirror it on the same tick the selection changes.
  //
  // Only `cwd` is consulted. `git_repo_root` is documented as null for non-git
  // workspaces and not-yet-backfilled history rows, so falling back to it would
  // read as "no workspace" for those sessions and blank a pane that was correct.
  const storedCwd = stored?.cwd?.trim() || ''

  if (storedCwd) {
    setCurrentCwdTransient(storedCwd)
    setWorkspaceCwdOwner(storedSessionId)
  } else {
    // Either a genuinely detached session, or a row outside the loaded sidebar
    // page (`stored` is undefined) — neither says anything about the workspace,
    // while `$currentCwd` still holds the previous conversation's folder.
    // Release so workspace-derived surfaces stop trusting it; `applyRuntimeInfo`
    // publishes the truth a moment later. The path is deliberately left in place
    // — clearing it collapses the workspace/review panes and drops file-tree
    // state on every switch.
    releaseWorkspaceCwdOwner()
  }

  // Same window, same reasoning: the branch is derived from the workspace, so
  // carrying the previous conversation's label across a switch is never right.
  setCurrentBranch('')
}
