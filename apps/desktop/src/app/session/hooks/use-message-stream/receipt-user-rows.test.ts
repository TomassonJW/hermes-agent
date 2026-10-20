import type { PersistedTurn } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { bindReceiptUserRows } from './receipt-user-rows'

const receipt: PersistedTurn = {
  row_ids: [110, 111],
  user_row_id: 110,
  final_assistant_row_id: 111,
  complete: true
}

const oldPrompt: ChatMessage = {
  id: 'user-old',
  role: 'user',
  parts: [{ type: 'text', text: 'Inspect this.' }]
}

const oldReply: ChatMessage = {
  id: 'assistant-stream-old',
  role: 'assistant',
  parts: [{ type: 'text', text: 'First response.' }],
  pending: false,
  interim: false,
  durableComplete: false
}

const newReply: ChatMessage = {
  id: 'assistant-stream-new',
  role: 'assistant',
  parts: [{ type: 'text', text: 'Background response.' }],
  pending: false,
  interim: false,
  rowId: 111,
  durableComplete: true,
  persistedTurn: receipt
}

describe('completion receipt ownership', () => {
  it.each(['assistant-stream-old', 'assistant-1700000000000'])(
    'never assigns a later background receipt to an earlier %s occurrence',
    id => {
      const messages = [oldPrompt, { ...oldReply, id }, newReply]

      expect(bindReceiptUserRows(messages, receipt)).toBe(messages)
      expect(oldPrompt.rowId).toBeUndefined()
    }
  )
  it('never binds an earlier unanswered receipt to the next background turn', () => {
    // The first completion lost its receipt. The background turn's opener did
    // not stream into this window, so only its final response is on screen.
    const messages = [oldPrompt, oldReply, newReply]

    expect(bindReceiptUserRows(messages, receipt)).toBe(messages)
    expect(oldPrompt.rowId).toBeUndefined()
  })

  it('still binds a queued prompt after an earlier turn lost its receipt', () => {
    const runningPrompt: ChatMessage = {
      id: 'user-next',
      role: 'user',
      parts: [{ type: 'text', text: 'Inspect this.' }]
    }

    const messages = [oldPrompt, oldReply, runningPrompt, newReply]
    const bound = bindReceiptUserRows(messages, receipt)

    expect(bound[0]).toBe(oldPrompt)
    expect(bound[2]).toEqual({ ...runningPrompt, rowId: 110 })
    expect(bound[3]).toBe(newReply)
  })
})
