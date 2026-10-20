/** Preserve only the local turn content not yet covered by durable history. */
import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText, sameAttachmentTurn, spliceOlderPreservedRows } from '@/lib/chat-messages'

import {
  isGatewaySystemMarker,
  isLiveTailRow,
  isStrictAnswerTextExtension,
  localPendingSupersedes,
  settledReplyOverEmptyShell,
  toolCallIdsOf,
  withAuthoritativeTurnState
} from './pending-turn-content'
import {
  acknowledgedTranscriptBoundary,
  conflictingTranscriptIdentity,
  transcriptRowIds
} from './pending-turn-identity'

/** Text of the response that follows a folded tool round, not the commentary before it. */
function lastFoldedResponseText(message: ChatMessage): string {
  let afterTool = false
  let text = ''

  for (const part of message.parts) {
    if (part.type === 'tool-call') {
      afterTool = true
      text = ''

      continue
    }

    if (afterTool && part.type === 'text') {
      text = textWithoutReferenceLines(part.text).trim()
    }
  }

  return afterTool ? text : ''
}

const textPartsOf = (message: ChatMessage) =>
  message.parts.flatMap(part => {
    const text = part.type === 'text' ? textWithoutReferenceLines(part.text).trim() : ''

    return text ? [text] : []
  })

const isPrompt = (message: ChatMessage) => message.role === 'user' && !isGatewaySystemMarker(message)

/**
 * The store provably holds this row under ids no newer than `bound`: a
 * hydrated row, a submit-acknowledged prompt, or a settled reply whose
 * completion receipt names its rows. A still-streaming row proves nothing.
 */
function storedAtOrBefore(message: ChatMessage, bound: number): boolean {
  if (bound <= 0 || message.pending === true) {
    return false
  }

  const finalRowId = message.persistedTurn?.final_assistant_row_id

  const ids = [
    ...transcriptRowIds(message),
    ...(message.persistedTurn?.row_ids ?? []),
    ...(typeof finalRowId === 'number' ? [finalRowId] : [])
  ].filter(id => Number.isSafeInteger(id) && id > 0)

  return ids.length > 0 && Math.max(...ids) <= bound
}

/** Context the gateway re-injects as a user row; it opens no turn of its own. */
const SYNTHETIC_CONTEXT_ROW = /^\[(?:CONTEXT COMPACTION|CONTEXT SUMMARY\])/

/** Hydrated rows are keyed `${storedTimestamp}-${index}-${role}`; a re-store keeps that timestamp. */
function hydratedRowTimestamp(message: ChatMessage): number | undefined {
  const stamp = /^(\d+(?:\.\d+)?)-\d+-[a-z-]+$/.exec(message.id)?.[1]

  return stamp === undefined ? undefined : Number(stamp)
}

const promptText = (message: ChatMessage) =>
  textWithoutReferenceLines(chatMessageText(message)).replace(/\s+/g, ' ').trim()

interface StoredTurnOpener {
  rowId: number
  text: string
  timestamp: number | undefined
}

/**
 * Turns the page carries whole: a stored prompt or background notice that a
 * committed reply follows. A synthetic context row (a compaction summary)
 * opens nothing.
 */
function completedStoredTurnOpeners(messages: ChatMessage[]): StoredTurnOpener[] {
  const openers: StoredTurnOpener[] = []
  let replied = false

  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]

    if (
      message.role === 'assistant' &&
      message.rowId !== undefined &&
      !isLiveTailRow(message) &&
      message.durableComplete !== false &&
      chatMessageText(message).trim()
    ) {
      replied = true

      continue
    }

    const text = promptText(message)

    if (
      replied &&
      (message.role === 'user' || message.role === 'system') &&
      message.rowId !== undefined &&
      text &&
      !SYNTHETIC_CONTEXT_ROW.test(text)
    ) {
      openers.push({ rowId: message.rowId, text, timestamp: hydratedRowTimestamp(message) })
    }
  }

  return openers
}

/**
 * Turns run one at a time. A settled local reply whose turn opened at a
 * stored prompt finished before any later turn began, so once the page
 * carries a later turn whole, the reply is history the page has moved past
 * (its stored copy sits above that turn, or further up the transcript).
 * A re-stored older prompt (a compaction keeps its timestamp under a new id)
 * and a repeat of the same prompt prove nothing.
 */
function closedBeforeStoredTurn(
  message: ChatMessage,
  index: number,
  previousMessages: ChatMessage[],
  openers: StoredTurnOpener[]
): boolean {
  if (message.role !== 'assistant' || message.pending === true || message.error || !openers.length) {
    return false
  }

  const prompt = previousMessages
    .slice(0, index)
    .findLast(candidate => candidate.role === 'user' && !isGatewaySystemMarker(candidate))

  const promptIds = prompt ? transcriptRowIds(prompt) : []

  if (!prompt || !promptIds.length) {
    return false
  }

  const promptRowId = Math.max(...promptIds)
  const text = promptText(prompt)
  const timestamp = hydratedRowTimestamp(prompt)

  const receiptRowIds = [...(message.persistedTurn?.row_ids ?? []), ...transcriptRowIds(message)].filter(
    rowId => Number.isSafeInteger(rowId) && rowId > 0
  )

  const ownNewest = Math.max(promptRowId, ...receiptRowIds)

  return openers.some(
    opener =>
      opener.rowId > ownNewest &&
      opener.text !== text &&
      (timestamp === undefined || opener.timestamp === undefined || opener.timestamp > timestamp)
  )
}

/**
 * Committed rows folding the local turn around `index`, for ANY turn, not just
 * the latest: rows sharing a tool call id with that turn, plus the assistant
 * rows after its prompt's durable twin. Only the latest turn may fall back to
 * position (everything after the last stored prompt).
 */
function committedFoldsOfLocalTurn(candidates: ChatMessage[], previous: ChatMessage[], index: number): ChatMessage[] {
  const start = previous.findLastIndex((row, at) => at < index && isPrompt(row))
  const end = previous.findIndex((row, at) => at > index && isPrompt(row))

  const turnToolIds = new Set(
    previous
      .slice(start + 1, end < 0 ? undefined : end)
      .flatMap(toolCallIdsOf)
      .filter(Boolean)
  )

  const owner = previous[start]
  const ownerRowIds = owner ? transcriptRowIds(owner) : []

  const anchor = owner
    ? candidates.findIndex(row => row.id === owner.id || transcriptRowIds(row).some(id => ownerRowIds.includes(id)))
    : -1

  const from = anchor >= 0 ? anchor : end < 0 ? candidates.findLastIndex(isPrompt) : candidates.length
  const until = candidates.findIndex((row, at) => at > from && isPrompt(row))
  const segment = new Set(candidates.slice(from + 1, until < 0 ? undefined : until))

  return candidates.filter(
    row =>
      row.role === 'assistant' &&
      !isLiveTailRow(row) &&
      (segment.has(row) || toolCallIdsOf(row).some(id => turnToolIds.has(id)))
  )
}

const hasWholeLines = (haystack: string, needle: string) => `\n${haystack}\n`.includes(`\n${needle}\n`)

/**
 * A fold carries sealed text verbatim as a text part, or inside Thinking when
 * the provider stored public commentary in `reasoning` (Codex Responses, #119716).
 */
const foldCarriesText = (fold: ChatMessage, text: string) =>
  fold.parts.some(part =>
    part.type === 'text'
      ? textWithoutReferenceLines(part.text).trim() === text
      : part.type === 'reasoning' && hasWholeLines(part.text, text)
  )

/**
 * History folds a tool-heavy turn into one bubble, while the live stream sealed
 * each interim segment and the final answer as bubbles of their own. A sealed
 * live bubble is that same occurrence when its turn's folds hold every tool
 * call it ran (durable identity) and its text, either verbatim (a sealed middle
 * segment, #119540) or as the final answer, equal or extended (#118670). This
 * holds with a partial or missing completion receipt, where full-bubble
 * equality sees neither.
 */
function durableFoldCoversLiveResponse(folds: ChatMessage[], live: ChatMessage): boolean {
  const liveToolIds = toolCallIdsOf(live)
  const sealed = live.pending !== true || live.interim === true

  if (!folds.length || (liveToolIds.length && (!sealed || liveToolIds.some(id => !id)))) {
    return false
  }

  const liveTexts = textPartsOf(live)

  const answer = liveToolIds.length
    ? lastFoldedResponseText(live)
    : textWithoutReferenceLines(chatMessageText(live)).trim()

  if (!answer && !liveTexts.length && !liveToolIds.length) {
    return false
  }

  const foldedToolIds = new Set(folds.flatMap(toolCallIdsOf))

  if (!liveToolIds.every(id => foldedToolIds.has(id))) {
    return false
  }

  if (sealed && liveTexts.every(text => folds.some(fold => foldCarriesText(fold, text)))) {
    return true
  }

  return (
    Boolean(answer) &&
    folds.some(fold => {
      const folded = lastFoldedResponseText(fold)

      return folded === answer || isStrictAnswerTextExtension(folded, answer)
    })
  )
}

function indexNextRoleOrdinals(remainingNext: ChatMessage[]): Map<string, ChatMessage> {
  const nextByRoleOrdinal = new Map<string, ChatMessage>()
  const nextRoleCounts = new Map<ChatMessage['role'], number>()

  for (const message of remainingNext) {
    if (isGatewaySystemMarker(message)) {
      continue
    }

    const ordinal = nextRoleCounts.get(message.role) ?? 0
    nextRoleCounts.set(message.role, ordinal + 1)
    nextByRoleOrdinal.set(`${message.role}:${ordinal}`, message)
  }

  return nextByRoleOrdinal
}

function liveOptimisticUserRun(previousMessages: ChatMessage[], acknowledgedLocalIndex: number): Set<ChatMessage> {
  const newestOptimisticUser = [...previousMessages]
    .reverse()
    .find(message => message.role === 'user' && message.id.startsWith('user-'))

  // A mid-turn redirect inserts its correction as a second optimistic user row
  // directly before the live reply, so one turn can own a contiguous RUN of
  // them. Preserving only the newest keeps the correction and drops the prompt
  // that started the turn. Widen to the run — but only the contiguous one: any
  // `user-*` row separated by an assistant reply is stale post-compression
  // history, which is what the newest-only rule exists to discard.
  const liveOptimisticUsers = new Set<ChatMessage>()

  if (newestOptimisticUser) {
    for (let index = previousMessages.indexOf(newestOptimisticUser); index > acknowledgedLocalIndex; index -= 1) {
      const candidate = previousMessages[index]

      if (candidate.role === 'user' && candidate.id.startsWith('user-')) {
        liveOptimisticUsers.add(candidate)

        continue
      }

      // Arrival-ordered mid-turn corrections sit BELOW the sealed live output
      // (#73793): a live-tail assistant row between the prompt and its
      // correction is still the same turn's run. Only a committed reply ends
      // it — that is the post-compression staleness the rule exists to catch.
      if (candidate.role === 'assistant' && isLiveTailRow(candidate)) {
        continue
      }

      break
    }
  }

  return liveOptimisticUsers
}

function pendingAssistantAlreadyStored(
  message: ChatMessage,
  lastStoredRowId: number,
  storedRowIds: Set<number>
): boolean {
  // Same for a reply whose completion receipt proved persistence: compaction
  // can re-insert it under a newer id, so absence here is not loss. #117867
  if (message.durableComplete === true && message.rowId !== undefined && message.rowId <= lastStoredRowId) {
    return true
  }

  // A turn that compressed mid-flight earns only a partial receipt, which
  // cannot vouch for the whole bubble. Once the store has moved past every
  // row it names and holds none of them, a later compaction rewrote the turn.
  const receiptRowIds = message.persistedTurn?.row_ids ?? []

  if (
    message.pending !== true &&
    receiptRowIds.length > 0 &&
    receiptRowIds.every(rowId => rowId <= lastStoredRowId && !storedRowIds.has(rowId))
  ) {
    return true
  }

  return false
}

export function preserveLocalPendingTurnMessages(
  nextMessages: ChatMessage[],
  previousMessages: ChatMessage[]
): ChatMessage[] {
  if (!previousMessages.length) {
    return nextMessages
  }

  const acknowledged = acknowledgedTranscriptBoundary(nextMessages, previousMessages)
  const remainingNext = nextMessages.slice(acknowledged.storedIndex + 1)
  const acknowledgedTurn = nextMessages.slice(Math.max(0, acknowledged.storedIndex))
  const storedRowIds = new Set(nextMessages.flatMap(transcriptRowIds))
  const lastStoredRowId = [...storedRowIds].reduce((last, rowId) => Math.max(last, rowId), 0)

  // The newest local row the store provably holds at or below the page's
  // newest row. Everything in front of it belongs to finished turns the page
  // has already moved past: a row the page lacks there is pagination or a
  // compaction rewrite, never unsent work. Appending it would paint an
  // earlier report, its narration or its prompt below the newer turn. The
  // horizon row itself is history only when its whole bubble is proven: a
  // partial receipt can still carry an unpersisted tail.
  const historyHorizon = previousMessages.findLastIndex(message => storedAtOrBefore(message, lastStoredRowId))

  const behindHistoryHorizon = (message: ChatMessage, index: number): boolean =>
    index < historyHorizon ||
    (index === historyHorizon && message.durableComplete !== false && message.persistedTurn?.complete !== false)

  const storedTurnOpeners = completedStoredTurnOpeners(nextMessages)

  const nextByRoleOrdinal = indexNextRoleOrdinals(remainingNext)

  const nextIds = new Set(nextMessages.map(message => message.id))
  const previousRoleCounts = new Map<ChatMessage['role'], number>()

  const liveOptimisticUsers = liveOptimisticUserRun(previousMessages, acknowledged.localIndex)

  // #121088: the acknowledged prompt's committed twin can sit anywhere in the
  // newly committed window — a compaction handoff or preserved-task notice
  // (synthetic user rows) may be NEWER than it, so the newest-only compare
  // misses the committed copy and the optimistic row is re-appended below the
  // whole refreshed turn. Dedupe against EVERY newly committed durable user
  // row, plus the newest user row as it was before (a rowId-less positional
  // hydration window keeps parity with the legacy compare). Every candidate
  // is identity-gated: a rowId-bearing optimistic row is never matched
  // against a committed row it provably is not, so a genuinely
  // unacknowledged repeat whose committed twin predates the acknowledged
  // boundary (and never enters this window) still survives.
  const newestAuthoritativeUser = [...remainingNext].reverse().find(message => message.role === 'user')

  const acknowledgedUserCandidates = remainingNext.filter(
    message =>
      message.role === 'user' &&
      !isGatewaySystemMarker(message) &&
      (message.rowId !== undefined || message === newestAuthoritativeUser)
  )

  const preserved: ChatMessage[] = []
  // Authoritative id → richer local pending row. Replacing (not appending)
  // avoids painting both the empty inflight shell and the full stream bubble.
  const replacements = new Map<string, ChatMessage>()
  let crossedUserBoundary = false
  let preservedLocalPrompt = false

  // A preserved prompt opens a DISTINCT occurrence only when an earlier
  // local reply is already covered by the acknowledged stored turn. With
  // no such anchor a truncated page can simply omit this very prompt.
  const followsCoveredReply = (index: number): boolean =>
    acknowledged.localIndex >= 0 &&
    previousMessages
      .slice(acknowledged.localIndex + 1, index)
      .some(
        local =>
          local.role === 'assistant' &&
          !local.pending &&
          !local.interim &&
          !local.error &&
          acknowledgedTurn.some(
            stored =>
              stored.role === 'assistant' &&
              !conflictingTranscriptIdentity(local, stored) &&
              textWithoutReferenceLines(chatMessageText(stored)) === textWithoutReferenceLines(chatMessageText(local))
          )
      )

  for (const [index, message] of previousMessages.entries()) {
    if (index <= acknowledged.localIndex || isGatewaySystemMarker(message)) {
      continue
    }

    crossedUserBoundary ||= message.role === 'user' || message.role === 'system'

    if (message.role === 'user' || message.role === 'system') {
      preservedLocalPrompt = false
    }

    // A second segment of the acknowledged turn can still be folded into its
    // final row. A new prompt closes that ownership; identical later replies
    // must not be consumed by the already-acknowledged prefix.
    const candidates = (crossedUserBoundary ? remainingNext : acknowledgedTurn).filter(
      candidate => !conflictingTranscriptIdentity(message, candidate)
    )

    const ordinal = previousRoleCounts.get(message.role) ?? 0
    previousRoleCounts.set(message.role, ordinal + 1)

    const isOptimisticUser = message.role === 'user' && message.id.startsWith('user-')

    const isPendingAssistant =
      message.role === 'assistant' && (message.pending === true || message.id.startsWith('assistant-stream-'))

    // A settled live-tail reply the backend has not committed yet is the only
    // copy of that reply and must survive a stale refreshed page (#121613).
    // Stream-id rows already enter through isPendingAssistant; this covers the
    // settled rows that do not (an interim id the completion settled onto, or
    // an appended `assistant-<ts>` bubble). Only the single newest row is
    // guarded: anything with a newer local row after it (stale compression
    // history, a superseded segment, the next turn's stream) belongs to the
    // reconcile paths. Interim, hidden, and persisted rows stay out for the
    // same reason: superseded interims, invisible rows, and fetchable rows
    // are not the live tail.
    //
    // A page that brings its own assistant row for this same reply ordinal has
    // rotated or compacted the transcript: that answer owns the slot, so the
    // cached row is superseded history rather than a live tail — resurrecting
    // it puts the pre-compression answer back on screen beside its
    // replacement. A stale page ends at the prompt (the reply is uncommitted),
    // so the ordinal stays empty there and the only copy still survives.
    const isSettledUnpersistedAssistant =
      index === previousMessages.length - 1 &&
      message.role === 'assistant' &&
      message.pending !== true &&
      message.interim !== true &&
      message.hidden !== true &&
      message.rowId === undefined &&
      chatMessageText(message).trim() !== '' &&
      (preservedLocalPrompt || !nextByRoleOrdinal.has(`assistant:${ordinal}`))

    if (!isOptimisticUser && !isPendingAssistant && !isSettledUnpersistedAssistant) {
      continue
    }

    // Same id already present: still prefer a strictly more complete local
    // pending body over an empty/stale shell that reused the stream id.
    if (nextIds.has(message.id)) {
      if (isOptimisticUser) {
        const existing = nextMessages.find(candidate => candidate.id === message.id)

        preservedLocalPrompt = Boolean(
          existing && transcriptRowIds(existing).length === 0 && followsCoveredReply(index)
        )
      }

      if (isPendingAssistant) {
        const existing = nextMessages.find(candidate => candidate.id === message.id)

        if (existing && localPendingSupersedes(message, existing)) {
          replacements.set(message.id, withAuthoritativeTurnState(message, existing))
        }
      }

      continue
    }

    if (isOptimisticUser && !liveOptimisticUsers.has(message)) {
      continue
    }

    // The submit receipt already proved this row was saved. If the newest
    // page has advanced beyond it, its absence is pagination, not unsent input.
    if (isOptimisticUser && message.rowId !== undefined && message.rowId <= lastStoredRowId) {
      continue
    }

    if (isPendingAssistant && pendingAssistantAlreadyStored(message, lastStoredRowId, storedRowIds)) {
      continue
    }

    const receiptRowIds = message.persistedTurn?.row_ids ?? []

    // If this window still owns an unacknowledged prompt, its answer is a
    // distinct occurrence. An older stored reply with identical prose (or
    // the same role ordinal) cannot prove this occurrence was persisted.
    if (
      preservedLocalPrompt &&
      message.role === 'assistant' &&
      transcriptRowIds(message).length === 0 &&
      receiptRowIds.length === 0 &&
      !behindHistoryHorizon(message, index)
    ) {
      preserved.push(message)

      continue
    }

    if (
      isOptimisticUser &&
      acknowledgedUserCandidates.some(
        candidate =>
          // #122079: the tolerant arm widens the TEXT compare only — it stays
          // inside the identity gate, so a rowId-bearing optimistic row is
          // never swallowed by a committed row it provably is not (a genuine
          // repeat of the same captioned paste). The rowId-less paste from
          // #120978 carries no identity and keeps matching tolerantly.
          !conflictingTranscriptIdentity(message, candidate) &&
          (textWithoutReferenceLines(chatMessageText(candidate)) ===
            textWithoutReferenceLines(chatMessageText(message)) ||
            sameAttachmentTurn(candidate, message))
      )
    ) {
      continue
    }

    const ordinalMatch = nextByRoleOrdinal.get(`${message.role}:${ordinal}`)

    const authoritative =
      ordinalMatch && !conflictingTranscriptIdentity(message, ordinalMatch) ? ordinalMatch : undefined

    // A settled stream row (`pending: false` after message.complete) whose reply
    // the authoritative transcript already carries under its committed id is
    // stale: ordinal pairing can't see it, because the commit shifted the row
    // one ordinal earlier, and re-appending it renders the same answer twice
    // (#70209). Only text-identical rows are dropped — a settled row the backend
    // has NOT committed yet is the only copy of that reply and must survive.
    if (
      (isPendingAssistant || isSettledUnpersistedAssistant) &&
      message.pending !== true &&
      candidates.some(
        candidate =>
          candidate.role === 'assistant' &&
          textWithoutReferenceLines(chatMessageText(candidate)) === textWithoutReferenceLines(chatMessageText(message))
      )
    ) {
      continue
    }

    if (authoritative) {
      if (isPendingAssistant) {
        // Keep the local pending row when it is the same reply further along
        // and the authoritative row is an empty projection shell or a prefix.
        // #75825. A settled final over a committed text-less tool shell is
        // the only copy of the answer (#123047).
        if (!localPendingSupersedes(message, authoritative) && !settledReplyOverEmptyShell(message, authoritative)) {
          continue
        }

        replacements.set(authoritative.id, withAuthoritativeTurnState(message, authoritative))

        continue
      }

      if (
        textWithoutReferenceLines(chatMessageText(authoritative)) ===
        textWithoutReferenceLines(chatMessageText(message))
      ) {
        continue
      }
    }

    // Ordinal pairing missed (the committed row shifted ordinal when history
    // was compacted / the authoritative list is shorter), yet the
    // authoritative transcript already carries this same reply under its
    // committed id. The #70209 guard above only covers SETTLED local rows
    // (`pending !== true`); a still-pending stream row that slips past
    // pairing falls through to `preserved.push` and renders the answer
    // twice — the reported A B C D E C D tail duplication.
    //
    // Three-way same-turn check against SETTLED authoritative rows only
    // (a live projection shell must not swallow the richer local row, see
    // the traces-only replacement test):
    //  1. identical answer text            -> authoritative already has it
    //  2. authoritative extends local text -> authoritative is the settled
    //     final version of the still-streaming local copy
    //  3. local extends authoritative text -> local is further along; replace
    //     the committed row with the richer body instead of appending
    if (isPendingAssistant) {
      const nextText = textWithoutReferenceLines(chatMessageText(message))

      const committedMatch = candidates.find(
        candidate =>
          candidate.role === 'assistant' &&
          !isLiveTailRow(candidate) &&
          (textWithoutReferenceLines(chatMessageText(candidate)) === nextText ||
            isStrictAnswerTextExtension(textWithoutReferenceLines(chatMessageText(candidate)), nextText))
      )

      if (committedMatch) {
        continue
      }

      const committedPrefix = candidates.find(
        candidate =>
          candidate.role === 'assistant' &&
          !isLiveTailRow(candidate) &&
          isStrictAnswerTextExtension(nextText, textWithoutReferenceLines(chatMessageText(candidate)))
      )

      if (committedPrefix) {
        // Keep the COMMITTED id (not the local stream id): the turn is
        // already in the authoritative transcript, so the merged row must
        // stay addressable as that durable row — a stream id would read as a
        // live row again next reconcile and re-enter this same path.
        replacements.set(committedPrefix.id, {
          ...withAuthoritativeTurnState(message, committedPrefix),
          id: committedPrefix.id
        })

        continue
      }
    }

    if (
      isPendingAssistant &&
      durableFoldCoversLiveResponse(committedFoldsOfLocalTurn(candidates, previousMessages, index), message)
    ) {
      continue
    }

    if (
      behindHistoryHorizon(message, index) ||
      closedBeforeStoredTurn(message, index, previousMessages, storedTurnOpeners)
    ) {
      continue
    }

    preserved.push(message)

    if (isOptimisticUser) {
      preservedLocalPrompt = followsCoveredReply(index)
    }
  }

  const withReplacements =
    replacements.size > 0 ? nextMessages.map(message => replacements.get(message.id) ?? message) : nextMessages

  // #120978: a kept run whose rowIds predate the whole hydrated page belongs
  // earlier — splice it in front of the first newer row instead of appending it
  // below the newest turn (non-qualifying runs keep the trailing behavior).
  return preserved.length ? spliceOlderPreservedRows(withReplacements, preserved) : withReplacements
}
