/** Reconcile an overlapping stored page in display order, retaining unmatched window occurrences. */
import { transcriptRowIds } from '@/app/session/hooks/use-session-actions/pending-turn-identity'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

export function durableRowIds(messages: ChatMessage[]): Set<number> {
  return new Set(messages.flatMap(message => (message.rowId === undefined ? [] : [message.rowId])))
}

function sharesDurableRow(first: ChatMessage[], second: ChatMessage[]): boolean {
  const rowIds = durableRowIds(first)

  return second.some(message => message.rowId !== undefined && rowIds.has(message.rowId))
}

interface PageSlot {
  message: ChatMessage
  /** Rows without a stored id (e.g. a page-local tool fold) that precede this row. */
  leading: ChatMessage[]
  /** Window rows the page does not carry that stay in front of this row. */
  before: ChatMessage[]
  /** Window rows the page does not carry that stay right after this row. */
  after: ChatMessage[]
  /** A window row already maps here; a later window copy is a duplicate. */
  claimed: boolean
  /** The leading rows came from the window, not from the page. */
  windowLeading?: boolean
}

interface WindowRow {
  message: ChatMessage
  leading: ChatMessage[]
  index: number
}

const normalizedText = (message: ChatMessage) => chatMessageText(message).replace(/\s+/g, ' ').trim()

/** Hydrated rows are keyed `${storedTimestamp}-${index}-${role}`; a re-store keeps that timestamp. */
function hydratedTimestamp(message: ChatMessage): string | undefined {
  return /^(\d+(?:\.\d+)?)-\d+-[a-z-]+$/.exec(message.id)?.[1]
}

/** The page bubble ends with this window bubble's text (a hydrated turn folds its narration in front). */
function endsWithText(fresh: string, local: string): boolean {
  return fresh === local || (fresh.endsWith(local) && /[\s.!?:;)\]]$/.test(fresh.slice(0, fresh.length - local.length)))
}

interface RowKey {
  id: string
  role: ChatMessage['role']
  text: string
  stamp: string | undefined
  settled: boolean
}

function rowKey(message: ChatMessage): RowKey {
  return {
    id: message.id,
    role: message.role,
    text: normalizedText(message),
    stamp: hydratedTimestamp(message),
    settled: !message.pending && !message.interim && !message.error
  }
}

/**
 * A compaction stores the rows it keeps again, under new ids, with the same
 * role, text and stored timestamp; the server then shows only the new copy.
 * The window can still hold the old copy, from an earlier page or from a
 * completion receipt. Equal prose is only a candidate: `restoredIdentity`
 * says whether the stored identity itself proves it.
 */
function sameRestoredRow(local: RowKey, fresh: RowKey): boolean {
  if (local.id === fresh.id) {
    return true
  }

  if (!local.settled || local.role !== fresh.role || !local.text) {
    return false
  }

  if (local.stamp !== undefined && fresh.stamp !== undefined && local.stamp !== fresh.stamp) {
    return false
  }

  return local.role === 'assistant' ? endsWithText(fresh.text, local.text) : local.text === fresh.text
}

/** Same hydrated id, or the same stored timestamp: the same stored row, not just the same words. */
function restoredIdentity(local: ChatMessage, fresh: ChatMessage): boolean {
  const stamp = hydratedTimestamp(local)

  return local.id === fresh.id || (stamp !== undefined && stamp === hydratedTimestamp(fresh))
}

/**
 * Sealed narration of a completed turn, still in the window as its own
 * bubble, that the page now carries inside the turn's hydrated fold. The
 * turn's receipt must name rows the fold absorbed; only the unstored
 * assistant bubbles right in front of the reply, whose words the fold holds,
 * give way.
 */
function withoutFoldedNarration(leading: ChatMessage[], reply: ChatMessage, fold: ChatMessage): ChatMessage[] {
  const receipt = reply.persistedTurn

  if (!receipt?.row_ids?.length) {
    return leading
  }

  // A partial receipt still names only committed rows. The fold must hold
  // one of them besides the reply itself: the row the narration was stored as.
  const turnRows = new Set(receipt.row_ids)
  const replyRows = new Set([...transcriptRowIds(reply), receipt.final_assistant_row_id])

  if (!transcriptRowIds(fold).some(id => turnRows.has(id) && !replyRows.has(id))) {
    return leading
  }

  let start = leading.length

  while (start > 0) {
    const message = leading[start - 1]

    if (message.role !== 'assistant' || message.rowId !== undefined || message.pending || message.error) {
      break
    }

    start--
  }

  const foldText = normalizedText(fold)

  const kept = leading.slice(start).filter(message => {
    const text = normalizedText(message)

    return !text || !foldText.includes(text)
  })

  return [...leading.slice(0, start), ...kept]
}

const toolCallIds = (message: ChatMessage) =>
  message.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))

/**
 * An unstored window row the page now stores: an optimistic prompt or a live
 * notice with the same words, or a live-streamed reply whose tool calls the
 * page row ran and whose words it carries. Any other local row without an
 * id is an unknown occurrence and is never retired on prose alone.
 */
function liveCopyOf(local: ChatMessage, fresh: ChatMessage): boolean {
  if (local.rowId !== undefined || local.pending || local.error || local.role !== fresh.role) {
    return false
  }

  const text = normalizedText(local)

  if (local.role === 'user' || local.role === 'system') {
    return Boolean(text) && text === normalizedText(fresh)
  }

  if (local.role !== 'assistant' || !(isLiveTailReplyId(local.id) || /^assistant-\d+$/.test(local.id))) {
    return false
  }

  const localTools = toolCallIds(local)
  const freshTools = new Set(toolCallIds(fresh))

  if (localTools.some(id => !id || !freshTools.has(id))) {
    return false
  }

  if (!text) {
    return localTools.length > 0
  }

  return (
    endsWithText(normalizedText(fresh), text) ||
    fresh.parts.some(part => part.type === 'text' && part.text.replace(/\s+/g, ' ').trim() === text)
  )
}

/**
 * The stored bubble already holds the whole local reply: every tool call it
 * ran and its full text as the bubble's ending. A partially receipted reply
 * can then carry no unpersisted suffix.
 */
function storedBubbleHoldsReply(stored: ChatMessage, local: ChatMessage): boolean {
  const text = normalizedText(local)
  const storedTools = new Set(toolCallIds(stored))

  return (
    Boolean(text) && toolCallIds(local).every(id => storedTools.has(id)) && endsWithText(normalizedText(stored), text)
  )
}

/** A later reply the store completed in full, after every row of this one. */
function closedByLaterStoredTurn(messages: ChatMessage[], index: number, rowIds: number[]): boolean {
  const newest = Math.max(0, ...rowIds)

  return messages
    .slice(index + 1)
    .some(
      later =>
        later.role === 'assistant' &&
        !later.pending &&
        !later.interim &&
        !later.error &&
        later.durableComplete === true &&
        later.persistedTurn?.complete === true &&
        (later.rowId ?? 0) > newest
    )
}

/**
 * Window rows the page stores again, matched in order. A stored bubble can
 * fold several live assistant bubbles (narration, then the reply), so those
 * may share one stored copy; a prompt or notice maps to one stored row each.
 */
function liveCopies(local: ChatMessage[], stored: ChatMessage[]): Map<ChatMessage, ChatMessage> {
  const copies = new Map<ChatMessage, ChatMessage>()
  let from = 0

  for (const message of local) {
    const found = stored.findIndex((candidate, index) => index >= from && liveCopyOf(message, candidate))

    if (found < 0) {
      continue
    }

    copies.set(message, stored[found])
    // A folded row may cover narration followed by its own final reply, but
    // never two completed replies from distinct occurrences. A new prompt
    // below a consumed reply must not let repeated prose reuse that row.
    from = message.role === 'assistant' && message.interim === true ? found : found + 1
  }

  return copies
}

/** Longest order-preserving pairing of window rows with the page rows that re-store them. */
function pairRestoredRows(local: ChatMessage[], fresh: ChatMessage[]): Map<ChatMessage, ChatMessage> {
  const pairs = new Map<ChatMessage, ChatMessage>()

  if (!local.length || !fresh.length) {
    return pairs
  }

  const localKeys = local.map(rowKey)
  const freshKeys = fresh.map(rowKey)
  const same = localKeys.map(left => freshKeys.map(right => sameRestoredRow(left, right)))
  const table = Array.from({ length: local.length + 1 }, () => new Uint32Array(fresh.length + 1))

  for (let i = local.length - 1; i >= 0; i--) {
    for (let j = fresh.length - 1; j >= 0; j--) {
      table[i][j] = same[i][j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1])
    }
  }

  let i = 0
  let j = 0

  while (i < local.length && j < fresh.length) {
    if (same[i][j] && table[i][j] === table[i + 1][j + 1] + 1) {
      pairs.set(local[i], fresh[j])
      i++
      j++
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      i++
    } else {
      j++
    }
  }

  return pairs
}

interface OverlapPage {
  refreshedIds: Set<string>
  refreshedBySource: Map<number, ChatMessage>
  slots: PageSlot[]
  slotByRowId: Map<number, PageSlot>
  slotByMessage: Map<ChatMessage, PageSlot>
  refreshedTrailing: ChatMessage[]
}

interface OverlapWindow {
  slotAt: (PageSlot | undefined)[]
  unmatched: WindowRow[]
  previousTrailing: ChatMessage[]
}

function indexRefreshedPage(refreshedTail: ChatMessage[]): OverlapPage {
  const refreshedIds = new Set(refreshedTail.map(message => message.id))
  const refreshedBySource = new Map<number, ChatMessage>()

  for (const message of refreshedTail) {
    for (const rowId of transcriptRowIds(message)) {
      refreshedBySource.set(rowId, message)
    }
  }

  const slots: PageSlot[] = []
  const slotByRowId = new Map<number, PageSlot>()
  const slotByMessage = new Map<ChatMessage, PageSlot>()
  let refreshedTrailing: ChatMessage[] = []

  for (const message of refreshedTail) {
    if (message.rowId === undefined) {
      refreshedTrailing.push(message)

      continue
    }

    const existing = slotByRowId.get(message.rowId)

    if (existing) {
      // A repeated stored id on the page: the later copy wins in place.
      slotByMessage.delete(existing.message)
      existing.message = message
      existing.leading = refreshedTrailing.length ? refreshedTrailing : existing.leading
      slotByMessage.set(message, existing)
    } else {
      const slot: PageSlot = { message, leading: refreshedTrailing, before: [], after: [], claimed: false }

      slots.push(slot)
      slotByRowId.set(message.rowId, slot)
      slotByMessage.set(message, slot)
    }

    refreshedTrailing = []
  }

  return { refreshedIds, refreshedBySource, slots, slotByRowId, slotByMessage, refreshedTrailing }
}

function mapWindowToPage(previous: ChatMessage[], page: OverlapPage): OverlapWindow {
  const { refreshedIds, refreshedBySource, slotByRowId } = page
  const slotAt: (PageSlot | undefined)[] = []
  const unmatched: WindowRow[] = []
  let pending: ChatMessage[] = []

  previous.forEach((message, index) => {
    const sourceIds = transcriptRowIds(message)
    const folded = sourceIds.length ? refreshedBySource.get(sourceIds[0]) : undefined

    // A completed live reply is addressed by its final row; hydration can
    // fold that row into an earlier tool bubble. Give both the fold's slot,
    // so the fresh copy replaces the live one without losing its leading rows.
    // Partial/live bubbles may still hold an uncommitted suffix. Equal words
    // alone never prove otherwise; a later turn the store completed does:
    // turns run one at a time, so this turn's stored rows are final, and the
    // stored bubble must then end with the whole local reply.
    const covered =
      !message.pending &&
      !message.interim &&
      !message.error &&
      folded?.role === message.role &&
      sourceIds.every(id => refreshedBySource.get(id) === folded) &&
      ((message.durableComplete !== false && message.persistedTurn?.complete !== false) ||
        (closedByLaterStoredTurn(previous, index, sourceIds) && storedBubbleHoldsReply(folded, message)))

    const rowId = covered ? (folded?.rowId ?? message.rowId) : message.rowId

    if (rowId === undefined) {
      // The fresh page's copy of an unstored row wins over the window's.
      if (!refreshedIds.has(message.id)) {
        pending.push(message)
      }

      return
    }

    const slot = slotByRowId.get(rowId)

    if (!slot) {
      unmatched.push({ message, leading: pending, index })
      pending = []

      return
    }

    // The fresh page replaces the row; keep the window's leading rows when
    // the page brought none of its own for it. A second window copy of the
    // same stored row is dropped with its leading rows.
    if (!slot.claimed) {
      slot.claimed = true

      if (!slot.leading.length && pending.length) {
        slot.leading = covered && folded ? withoutFoldedNarration(pending, message, folded) : pending
        slot.windowLeading = slot.leading.length > 0
      }
    }

    slotAt[index] = slot
    pending = []
  })

  const previousTrailing = pending

  return { slotAt, unmatched, previousTrailing }
}

function retireRestoredWindowRows(previous: ChatMessage[], page: OverlapPage, window: OverlapWindow): Set<ChatMessage> {
  const { slots, slotByMessage, refreshedBySource } = page
  const { slotAt, unmatched } = window
  // Old copies of re-stored rows sit next to the shared rows: in front of
  // them when the page re-stored the turn the window ends with, after them
  // when it re-stored rows it now shows at their old place.
  // A row whose own stored ids are still on the page is not an old copy: the
  // page holds that very row (possibly folded), and a partial receipt there
  // can still carry an unpersisted suffix.
  const unclaimed = slots.filter(slot => !slot.claimed).map(slot => slot.message)
  // Narration without stored ids does not bound the number of stored rows
  // a compaction may have re-written. Check every unmatched stored row;
  // identity/anchor proof below, not distance, decides which copy retires.
  const candidates = unmatched.filter(row => !transcriptRowIds(row.message).some(id => refreshedBySource.has(id)))

  const pairs = pairRestoredRows(
    candidates.map(row => row.message),
    unclaimed
  )

  const rowByMessage = new Map(candidates.map(row => [row.message, row]))

  const storedNeighbour = (index: number, step: -1 | 1): number => {
    let at = index + step

    while (at >= 0 && at < previous.length && previous[at].rowId === undefined) {
      at += step
    }

    return at >= 0 && at < previous.length ? at : -1
  }

  // Only page rows the window does not hold between two slots.
  const onlyUnclaimedBetween = (from: number, to: number) => slots.slice(from + 1, to).every(slot => !slot.claimed)

  // Equal words alone never retire a stored row: another occurrence can say
  // the same thing. The pair must also sit where the old copy sat, next to a
  // row both sides already share with nothing but rows the window never held
  // in between, in stored order, and the page copy must be the newer store.
  const anchored = (row: WindowRow, slot: PageSlot): boolean => {
    const at = slots.indexOf(slot)
    const rowId = row.message.rowId ?? 0
    const before = storedNeighbour(row.index, -1)
    const after = storedNeighbour(row.index, 1)
    const leftSlot = before >= 0 ? slotAt[before] : undefined
    const rightSlot = after >= 0 ? slotAt[after] : undefined
    const leftAt = leftSlot ? slots.indexOf(leftSlot) : -1
    const rightAt = rightSlot ? slots.indexOf(rightSlot) : -1

    if ((slot.message.rowId ?? 0) <= rowId) {
      return false
    }

    const leftHeld =
      leftAt >= 0 && leftAt < at && onlyUnclaimedBetween(leftAt, at) && rowId > (previous[before].rowId ?? Infinity)

    const rightHeld = rightAt > at && onlyUnclaimedBetween(at, rightAt) && rowId < (previous[after].rowId ?? 0)

    return leftHeld || rightHeld
  }

  const retired = new Set<ChatMessage>()

  const retire = (row: WindowRow, fresh: ChatMessage) => {
    const slot = slotByMessage.get(fresh)

    if (!slot || slot.claimed) {
      return false
    }

    const freshText = normalizedText(fresh)

    slot.claimed = true

    if (!slot.leading.length) {
      slot.leading = row.leading.filter(message => !freshText.includes(normalizedText(message)))
      slot.windowLeading = slot.leading.length > 0
    }

    slotAt[row.index] = slot
    retired.add(row.message)

    return true
  }

  for (const [message, fresh] of pairs) {
    const row = rowByMessage.get(message)

    if (row && restoredIdentity(message, fresh)) {
      retire(row, fresh)
    }
  }

  for (let changed = true; changed;) {
    changed = false

    for (const [message, fresh] of pairs) {
      const row = rowByMessage.get(message)
      const slot = slotByMessage.get(fresh)

      if (row && slot && !retired.has(message) && anchored(row, slot)) {
        changed = retire(row, fresh) || changed
      }
    }
  }

  return retired
}

function placeUnmatchedWindowRows(slots: PageSlot[], window: OverlapWindow, retired: Set<ChatMessage>): ChatMessage[] {
  const { slotAt, unmatched } = window
  const first = slotAt.findIndex(Boolean)
  const last = slotAt.findLastIndex(Boolean)
  const anchorSlot = first < 0 ? slots.length : slots.indexOf(slotAt[first] as PageSlot)
  const storedTrailing: ChatMessage[] = []
  let headSlot = 0

  for (const row of unmatched) {
    if (retired.has(row.message)) {
      continue
    }

    const rows = [...row.leading, row.message]

    if (first < 0 || row.index < first) {
      // Older than the page: in front of the first page row stored after it,
      // never past the page row the window shows after it.
      const rowId = row.message.rowId ?? 0
      const later = slots.findIndex((slot, at) => at < anchorSlot && (slot.message.rowId ?? 0) > rowId)

      headSlot = Math.max(headSlot, later < 0 ? anchorSlot : later)

      if (headSlot < slots.length) {
        slots[headSlot].before.push(...rows)
      } else {
        storedTrailing.push(...rows)
      }
    } else if (row.index > last) {
      // Newer than the page (stored after it was read).
      storedTrailing.push(...rows)
    } else {
      // Between rows the page carries: keep it next to the row it followed.
      let at = row.index - 1

      while (at >= 0 && !slotAt[at]) {
        at--
      }

      ;(slotAt[at] as PageSlot).after.push(...rows)
    }
  }

  return storedTrailing
}

function hoist(hoisted: Map<PageSlot, ChatMessage[]>, slot: PageSlot, rows: ChatMessage[]): void {
  if (rows.length) {
    hoisted.set(slot, [...(hoisted.get(slot) ?? []), ...rows])
  }
}

function hoistLeadingLiveCopies(page: OverlapPage): Map<PageSlot, ChatMessage[]> {
  const { slots, refreshedIds } = page
  // Unstored window rows in front of a shared row that the page now stores
  // just in front of it, among the rows the window never held, are those
  // stored copies: the window's give way, and whatever the window showed
  // before such a copy stays in front of it.
  const hoisted = new Map<PageSlot, ChatMessage[]>()

  slots.forEach((slot, at) => {
    if (!slot.windowLeading) {
      return
    }

    const gapOwner = new Map<ChatMessage, PageSlot>()
    const gap: ChatMessage[] = []

    for (let back = at - 1; back >= 0 && !slots[back].claimed; back--) {
      const rows = [...slots[back].leading, slots[back].message]

      gap.unshift(...rows)
      rows.forEach(row => gapOwner.set(row, slots[back]))
    }

    const leading = slot.leading.filter(message => !refreshedIds.has(message.id))
    const copies = liveCopies(leading, gap)
    let buffer: ChatMessage[] = []

    for (const message of leading) {
      const copy = copies.get(message)

      if (copy) {
        hoist(hoisted, gapOwner.get(copy) as PageSlot, buffer)
        buffer = []
      } else {
        buffer.push(message)
      }
    }

    slot.leading = buffer
  })

  return hoisted
}

function retainUnstoredWindowTail(
  previous: ChatMessage[],
  page: OverlapPage,
  window: OverlapWindow,
  storedTrailing: ChatMessage[]
): ChatMessage[] {
  const { slots, refreshedTrailing } = page
  const { previousTrailing, slotAt } = window
  // The window's unstored tail (a reply that finished without a receipt, a
  // prompt not yet acknowledged) is what the page now stores after the last
  // row both share, possibly behind rows the window never saw (a background
  // notice). Drop those copies, in stored order; keep anything still
  // streaming or that the page does not show there.
  let windowTail = previousTrailing
  const lastStored = previous.findLastIndex(message => message.rowId !== undefined)
  const tailSlot = lastStored >= 0 ? slotAt[lastStored] : undefined

  if (tailSlot && !storedTrailing.length && previousTrailing.length) {
    const newer = slots
      .slice(slots.indexOf(tailSlot) + 1)
      .flatMap(slot => (slot.claimed ? [] : [...slot.leading, slot.message]))
      .concat(refreshedTrailing)

    const copies = liveCopies(previousTrailing, newer)

    windowTail = previousTrailing.filter(message => !copies.has(message))
  }

  return windowTail
}

function hoistFinishedWindowReplies(slots: PageSlot[], hoisted: Map<PageSlot, ChatMessage[]>): void {
  // A finished reply the window holds without a stored id, in front of a
  // shared row, belongs to an earlier turn than that row. The page rows
  // stored just in front of the shared row that the window never saw (a
  // background notice, the next prompt) open the shared row's turn, so the
  // finished reply and what led up to it go in front of them.
  const finishedReply = (message: ChatMessage) =>
    message.role === 'assistant' &&
    !message.pending &&
    !message.interim &&
    !message.error &&
    Boolean(normalizedText(message))

  slots.forEach((slot, at) => {
    const cut = slot.windowLeading ? slot.leading.findLastIndex(finishedReply) + 1 : 0

    if (!cut) {
      return
    }

    let start = at

    while (start > 0 && !slots[start - 1].claimed) {
      start--
    }

    if (start === at) {
      return
    }

    hoist(hoisted, slots[start], slot.leading.slice(0, cut))
    slot.leading = slot.leading.slice(cut)
  })
}

/**
 * Merge for a page that overlaps the window but does not anchor in front of
 * it. The page is the newest stretch of the stored transcript, in the
 * server's display order, and that order is kept as is: a compaction stores
 * the rows it keeps again under new, larger ids at their old place, so
 * sorting by stored id would paint an earlier report under the newer turn.
 *
 * A window row the page does not carry is either older than the page (it
 * stays in front of the page row it preceded), newer than the page (it stays
 * after it), or an old copy of a row the page re-stored (it gives way). A row
 * with no stored id travels with the next stored row after it, so a
 * page-local fold stays in front of the row it preceded.
 */
export function mergeOverlappingTail(previous: ChatMessage[], refreshedTail: ChatMessage[]): ChatMessage[] {
  // Compaction, rewind, or a different session arrives as new stored ids.
  // This function is the path that puts that page on screen.
  if (!sharesDurableRow(previous, refreshedTail)) {
    return refreshedTail
  }

  const page = indexRefreshedPage(refreshedTail)
  const window = mapWindowToPage(previous, page)
  const retired = retireRestoredWindowRows(previous, page, window)
  const storedTrailing = placeUnmatchedWindowRows(page.slots, window, retired)
  const hoisted = hoistLeadingLiveCopies(page)
  const windowTail = retainUnstoredWindowTail(previous, page, window, storedTrailing)

  hoistFinishedWindowReplies(page.slots, hoisted)

  const { slots, refreshedTrailing } = page

  const stored = slots.flatMap(slot => [
    ...slot.before,
    ...(hoisted.get(slot) ?? []),
    ...slot.leading,
    slot.message,
    ...slot.after
  ])

  return [...stored, ...storedTrailing, ...refreshedTrailing, ...windowTail]
}
