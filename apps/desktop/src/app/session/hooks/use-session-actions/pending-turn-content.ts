/** Content and identity comparisons shared by resume and pending-turn preservation. */
import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

/** Reasoning / tool-call parts that the gateway inflight dump cannot express. */
export function hasStructuralParts(message: ChatMessage): boolean {
  return message.parts.some(part => part.type === 'reasoning' || part.type === 'tool-call')
}

/**
 * A live-turn row — the gateway's text-only `inflight` projection, a
 * still-streaming local bubble, or an interim row sealed inside the running
 * turn — as opposed to a committed transcript row.
 */
export function isLiveTailRow(message: ChatMessage): boolean {
  return message.pending === true || isLiveTailReplyId(message.id) || message.interim === true
}

/**
 * True when `next` is a pure forward extension of the previous *answer* text.
 * Empty previous answer never accepts a dump as an extension — that is how the
 * mid-turn inflight flat dump used to sandwich structured rows (#76444).
 */
export function isStrictAnswerTextExtension(next: string, previous: string): boolean {
  const n = next.trim()
  const p = previous.trim()

  if (!p || !n) {
    return false
  }

  return n.startsWith(p)
}

/**
 * Carry the durable row id and reactions from a same-turn `previous` row onto
 * `next` when it lacks them — reactions are keyed by row id, so they travel
 * together. Returns `next` itself when there is nothing to carry, else a NEW
 * object (the runtime repository caches normalized messages by identity).
 */
export function carryRowIdentity(next: ChatMessage, previous: ChatMessage): ChatMessage {
  const rowId = next.rowId === undefined ? previous.rowId : undefined
  const reactions = next.reactions === undefined && previous.reactions?.length ? previous.reactions : undefined

  if (rowId === undefined && !reactions) {
    return next
  }

  return {
    ...next,
    ...(rowId !== undefined ? { rowId } : {}),
    ...(reactions ? { reactions: [...reactions] } : {})
  }
}

/**
 * Keep the local tail of a turn while a reconnect hydrates an older server
 * projection. The user's optimistic row exists before prompt.submit persists
 * it, and the pending assistant row exists before message.complete commits it;
 * dropping either makes an accepted turn appear to vanish during transport
 * churn.
 *
 * A lagging projection can be behind by one live turn, never a whole local
 * history window. Preserve only the newest optimistic user row: compression
 * rewrites past context, so older `user-*` rows in a warm cache are stale
 * history, not in-flight work. The latest authoritative user confirms whether
 * that tail has persisted. An authoritative assistant at the same ordinal
 * supersedes the local stream only when it is at least as complete; an empty
 * or lagging inflight shell must not discard a fuller local pending reply
 * (#75825).
 *
 * Gateway bookkeeping markers (the model-switch / personality notices written
 * by tui_gateway/server.py) are persisted as role=user but are not user turns.
 * They must not take part in ordinal pairing on either side: a stored marker
 * between two real user turns shifts every later user ordinal, so the optimistic
 * row misses its committed copy and is appended a second time at the end of the
 * transcript — the duplicated user bubble of #67603.
 */
export const isGatewaySystemMarker = (message: ChatMessage): boolean =>
  message.role === 'user' && chatMessageText(message).trimStart().startsWith('[System:')

/**
 * Does the row carry anything a viewer would miss — streamed answer text, or
 * the reasoning / tool-call structure the gateway's flat dump cannot express?
 * An empty inflight shell carries none of it.
 */
const hasStreamedContent = (message: ChatMessage): boolean =>
  chatMessageText(message).trim().length > 0 || hasStructuralParts(message)

/**
 * #80151: the store's flat projection and the streamed parts join the same
 * segments with different separators (blank lines around folded tool rounds,
 * reference lines), so byte-prefix pairing misses the same turn. Compare the
 * answer text with reference lines stripped and separators folded away.
 */
const foldAnswerTextForCompare = (text: string): string => textWithoutReferenceLines(text).replace(/\s+/g, '')

/**
 * #80151: may a still-pending local stream claim a COMMITTED row whose answer
 * text the stream provably holds? With the turn proven by shared tool-call
 * ids (call ids are unique to the turn), the committed text is the mid-turn
 * segment the store flushed — it must be one of the stream's own folded text
 * segments, or a prefix of one (a mid-segment flush). Without that proof the
 * whole folded answer must strictly extend the committed text, mirroring the
 * live-shell prefix rule. An empty-text committed row is never claimable
 * here: empty prose carries no identity (#114543); the settled-final shell
 * case is preserveLocalPendingTurnMessages' own rule (#123047).
 */
const streamedAnswerHoldsCommittedText = (local: ChatMessage, authoritative: ChatMessage): boolean => {
  const foldedAuthoritative = foldAnswerTextForCompare(chatMessageText(authoritative))

  if (!foldedAuthoritative.length) {
    return false
  }

  const authoritativeToolIds = toolCallIdsOf(authoritative)

  if (authoritativeToolIds.length > 0 && authoritativeToolIds.every(id => toolCallIdsOf(local).includes(id))) {
    const localSegments = local.parts.flatMap(part =>
      part.type === 'text' ? [foldAnswerTextForCompare(part.text)] : []
    )

    return localSegments.some(segment => segment.startsWith(foldedAuthoritative))
  }

  const foldedLocal = foldAnswerTextForCompare(chatMessageText(local))

  return foldedLocal.length > foldedAuthoritative.length && foldedLocal.startsWith(foldedAuthoritative)
}

/**
 * May the cached local row stand in for this authoritative assistant?
 *
 * Only for a live projection of the SAME reply that the local copy is further
 * along on: an empty shell, or text the local row strictly extends. Comparing
 * lengths alone lets an unrelated (merely longer) local row hijack the ordinal
 * — or the stream id — of a genuine stored reply. A retained failure snapshot
 * (`inflight.error`, projected with empty text) is never a shell: repainting it
 * from the local partial would hide the error and mark the turn healthy again.
 *
 * #80151: a COMMITTED row at the same ordinal can be this turn's mid-turn
 * partial — the backend persists segments while the turn runs, so switching
 * chats mid-stream and back hydrates the store's flat projection next to the
 * still-streaming local copy, and that projection need not be a byte prefix of
 * the streamed text. Only a still-PENDING local row may claim it, and only
 * with same-turn proof: every tool-call id the committed row names also
 * streams locally (call ids are unique to the turn, so a different turn at
 * the same ordinal — a resent prompt answered earlier — cannot pass), or the
 * committed answer text is a strict folded prefix of the streamed text. In
 * both arms the streamed copy must cover the committed text, so a complete
 * earlier answer is never traded for a shorter local row.
 */
export const localPendingSupersedes = (local: ChatMessage, authoritative: ChatMessage): boolean => {
  if (local.role !== 'assistant' || !isLiveTailRow(local)) {
    return false
  }

  if (authoritative.error) {
    return false
  }

  const authoritativeText = chatMessageText(authoritative).trim()

  if (isLiveTailRow(authoritative)) {
    if (!authoritativeText.length) {
      return hasStreamedContent(local)
    }

    const localText = chatMessageText(local).trim()

    return localText.length > authoritativeText.length && isStrictAnswerTextExtension(localText, authoritativeText)
  }

  if (local.pending !== true) {
    return false
  }

  return hasStreamedContent(local) && streamedAnswerHoldsCommittedText(local, authoritative)
}

const answerText = (message: ChatMessage) => textWithoutReferenceLines(chatMessageText(message)).trim()

/**
 * A committed tool round with no prose (`content = ''`, `tool_calls` set)
 * hydrates as a text-less shell. When no later prose row folds into it, the
 * settled local final bubble is the only copy of the answer the user watched
 * stream, and dropping it for the shell deletes the reply on sync (#123047).
 * Interim narration and structural-only local rows never claim the slot, and
 * a retained error row is never a shell. Kept apart from
 * `localPendingSupersedes`, whose stricter live-only rule also gates resume.
 */
export const settledReplyOverEmptyShell = (local: ChatMessage, authoritative: ChatMessage): boolean =>
  local.role === 'assistant' &&
  local.pending !== true &&
  local.interim !== true &&
  !local.error &&
  authoritative.role === 'assistant' &&
  !isLiveTailRow(authoritative) &&
  !authoritative.error &&
  answerText(authoritative).length === 0 &&
  answerText(local).length > 0

/**
 * Take the cached row's content, but never its liveness. The renderer holds the
 * only copy of the streamed parts; the gateway remains the authority on whether
 * the turn is still running and on durable row identity — so a settled shell
 * must not repaint the reply as perpetually streaming.
 */
export const withAuthoritativeTurnState = (local: ChatMessage, authoritative: ChatMessage): ChatMessage =>
  carryRowIdentity({ ...local, pending: authoritative.pending === true }, authoritative)

export const toolCallIdsOf = (message: ChatMessage) =>
  message.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))
