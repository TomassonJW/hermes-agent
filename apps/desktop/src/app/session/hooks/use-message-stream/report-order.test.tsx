/**
 * Report order after a transcript refresh.
 *
 * A refresh puts the newest stored page on screen through the production
 * chain: extendRefreshPageToOverlap, graftRefreshedTailOntoBackfill, then the
 * preserveLocal* passes. Whatever the window streamed live, lost (a completion
 * receipt, a submit acknowledgement) or saw re-stored by a compaction, the
 * screen must read like a forced reload: the newest stretch of the stored
 * transcript, in stored order, each row once. A second refresh moves nothing.
 */
import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { extendRefreshPageToOverlap, graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import {
  type ChatMessage,
  chatMessageText,
  preserveLocalAssistantErrors,
  preserveLocalSystemNotices,
  toChatMessages
} from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { preserveLocalPendingTurnMessages } from '../use-session-actions/utils'

import { renderMessageStream } from './test-harness'

interface Scenario {
  /** Completion receipt of the first turn: full, partial (a compaction ran), or lost with its event. */
  receipt: 'full' | 'partial' | 'lost'
  /** Completion receipt of the following turn. */
  nextReceipt: 'full' | 'lost'
  /** What opened the following turn. */
  next: 'delegation' | 'process' | 'prompt'
  /** The window streamed the following reply live. */
  sawNext: boolean
  /** The store compacted after the first turn and re-stored the rows it kept under new ids. */
  compaction: boolean
  /** The first turn streamed narration before its tool call. */
  narration: boolean
  /** Where the refreshed page starts. */
  page: 'session' | 'first-turn' | 'mid-first-turn' | 'next-turn'
  /** Reading older pages to reach the window: works, finds nothing, or fails. */
  olderPages: 'works' | 'empty' | 'fails'
  /** How the window holds its prompts: hydrated, optimistic, or optimistic and acknowledged with a row id. */
  prompts: 'hydrated' | 'optimistic' | 'acknowledged'
}

const COMPACTION = '[CONTEXT COMPACTION'

const receiptFor = (kind: 'full' | 'partial' | 'lost', rowIds: number[]) =>
  kind === 'lost'
    ? undefined
    : {
        row_ids: rowIds,
        user_row_id: rowIds[0],
        final_assistant_row_id: rowIds[rowIds.length - 1],
        complete: kind === 'full'
      }

const optimisticPrompt = (id: string, text: string, rowId?: number) =>
  ({
    id,
    role: 'user',
    parts: [{ type: 'text', text }],
    ...(rowId === undefined ? {} : { rowId })
  }) as unknown as ChatMessage

/** What the screen reads: one atom per row, assistant bubbles split into sentences so folds compare equal. */
function screenOf(messages: ChatMessage[]): string[] {
  const atoms: string[] = []

  for (const message of messages) {
    const text = chatMessageText(message).replace(/\s+/g, ' ').trim()

    if (!text || text.startsWith(COMPACTION)) {
      continue
    }

    if (message.role !== 'assistant') {
      atoms.push(`${message.role}:${text}`)

      continue
    }

    for (const sentence of text.match(/[^.]+\.?/g) ?? []) {
      if (sentence.trim()) {
        atoms.push(`assistant:${sentence.trim()}`)
      }
    }
  }

  return atoms
}

let sessionSeq = 0

async function refreshTwice(scenario: Scenario) {
  const sessionId = `report-order-${++sessionSeq}`
  const h = renderMessageStream(sessionId)

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => h.handleEvent({ type, payload, session_id: sessionId }))

  const setWindow = (messages: ChatMessage[]) => h.states.set(sessionId, { ...h.state(sessionId), messages })

  const earlier: SessionMessage[] = [
    { id: 90, role: 'user', content: 'Earlier prompt.', timestamp: 90 },
    { id: 91, role: 'assistant', content: 'Earlier reply.', timestamp: 91 }
  ]

  const firstTurn: SessionMessage[] = [
    { id: 100, role: 'user', content: 'Inspect this.', timestamp: 100 },
    {
      id: 101,
      role: 'assistant',
      content: scenario.narration ? 'Narration one.' : '',
      timestamp: 101,
      tool_calls: [{ id: 'call-0', type: 'function', function: { name: 'terminal', arguments: '{"command":"pwd"}' } }]
    },
    { id: 102, role: 'tool', content: 'done', tool_call_id: 'call-0', tool_name: 'terminal', timestamp: 102 },
    { id: 103, role: 'assistant', content: 'First response.', timestamp: 103 }
  ]

  setWindow(toChatMessages(earlier))
  setWindow([
    ...h.state(sessionId).messages,
    ...(scenario.prompts === 'hydrated'
      ? toChatMessages([firstTurn[0]])
      : [optimisticPrompt('user-1', 'Inspect this.', scenario.prompts === 'acknowledged' ? 100 : undefined)])
  ])

  await send('message.start')

  if (scenario.narration) {
    await send('message.delta', { text: 'Narration one.' })
    await send('message.interim', { text: 'Narration one.', already_streamed: true })
  }

  await send('tool.start', { name: 'terminal', tool_id: 'call-0', args: { command: 'pwd' } })
  await send('tool.complete', { name: 'terminal', tool_id: 'call-0', result: 'done' })
  await send('message.delta', { text: 'First response.' })
  await send('message.complete', {
    text: 'First response.',
    status: 'complete',
    persisted_turn: receiptFor(scenario.receipt, [100, 101, 102, 103])
  })

  let stored: SessionMessage[] = [...earlier, ...firstTurn]
  let nextId = 110

  if (scenario.compaction) {
    // As in state.db: the summary first, then the kept rows stored again under new ids.
    const summary: SessionMessage = {
      id: 200,
      role: 'user',
      content: `${COMPACTION} — REFERENCE ONLY] Earlier turns were compacted.`,
      timestamp: 104
    }

    stored = [...earlier, summary, ...firstTurn.map((message, index) => ({ ...message, id: 201 + index }))]
    nextId = 210
  }

  const openers: Record<Scenario['next'], SessionMessage> = {
    delegation: {
      id: nextId,
      role: 'user',
      content: '[ASYNC DELEGATION BATCH COMPLETE — example]',
      timestamp: nextId,
      display_kind: 'async_delegation_complete'
    },
    process: {
      id: nextId,
      role: 'user',
      content: '[Background process proc_1 finished with exit code 0]',
      timestamp: nextId,
      display_kind: 'process_complete'
    },
    prompt: { id: nextId, role: 'user', content: 'Next question.', timestamp: nextId }
  }

  const opener = openers[scenario.next]

  if (scenario.next === 'prompt') {
    setWindow([
      ...h.state(sessionId).messages,
      scenario.prompts === 'hydrated'
        ? toChatMessages([opener])[0]
        : optimisticPrompt('user-2', 'Next question.', scenario.prompts === 'acknowledged' ? nextId : undefined)
    ])
  }

  if (scenario.sawNext) {
    await send('message.start')
    await send('message.delta', { text: 'Background response.' })
    await send('message.complete', {
      text: 'Background response.',
      status: 'complete',
      persisted_turn: receiptFor(scenario.nextReceipt, [nextId, nextId + 1])
    })
  }

  stored = [
    ...stored,
    opener,
    { id: nextId + 1, role: 'assistant', content: 'Background response.', timestamp: nextId + 1 }
  ]

  const pageStart = {
    session: 0,
    'first-turn': stored.findIndex(message => message.content === 'Inspect this.'),
    'mid-first-turn': stored.findIndex(message => message.role === 'tool'),
    'next-turn': stored.findIndex(message => message.id === nextId)
  }[scenario.page]

  const screens: string[][] = []
  let window = h.state(sessionId).messages

  for (let refresh = 0; refresh < 2; refresh++) {
    // Older pages, two stored rows at a time, like getOlderSessionMessages.
    let cursor = pageStart

    const readOlderPage = async (): Promise<ChatMessage[]> => {
      if (scenario.olderPages === 'fails') {
        throw new Error('network')
      }

      if (scenario.olderPages === 'empty' || cursor <= 0) {
        return []
      }

      const from = Math.max(0, cursor - 2)
      const rows = stored.slice(from, cursor)

      cursor = from

      return toChatMessages(rows)
    }

    const page = await extendRefreshPageToOverlap(toChatMessages(stored.slice(pageStart)), window, readOlderPage)
    const before = window

    window = preserveLocalSystemNotices(
      preserveLocalAssistantErrors(
        preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(page, before), before),
        before
      ),
      before
    )
    screens.push(screenOf(window))
  }

  cleanup()

  return {
    truth: screenOf(toChatMessages(stored)),
    // Every row actually returned by the server must survive the merge. A
    // valid suffix alone would also accept silently dropping most of a page.
    required: screenOf(toChatMessages(stored.slice(pageStart))),
    screens
  }
}

/** The newest stretch of the stored transcript, in stored order, each row once. Older rows come back on scroll. */
const readsLikeAReload = (screen: string[], truth: string[], required: string[]) =>
  screen.length > 0 &&
  screen.length >= required.length &&
  JSON.stringify(screen) === JSON.stringify(truth.slice(truth.length - screen.length))

const describeScenario = (scenario: Scenario) =>
  Object.entries(scenario)
    .map(([factor, value]) => `${factor}=${value}`)
    .join(' ')

type ScenarioFactors = { [K in keyof Scenario]: readonly Scenario[K][] }

function allScenarios(): Scenario[] {
  const factors: ScenarioFactors = {
    receipt: ['full', 'partial', 'lost'],
    nextReceipt: ['full', 'lost'],
    next: ['delegation', 'process', 'prompt'],
    sawNext: [true, false],
    compaction: [false, true],
    narration: [false, true],
    page: ['session', 'first-turn', 'mid-first-turn', 'next-turn'],
    olderPages: ['works', 'empty', 'fails'],
    prompts: ['hydrated', 'optimistic', 'acknowledged']
  }

  // Every factor is visited once; each final prefix has all Scenario fields.
  return (Object.keys(factors) as (keyof Scenario)[]).reduce<Partial<Scenario>[]>(
    (prefixes, key) => prefixes.flatMap(prefix => factors[key].map(value => ({ ...prefix, [key]: value }))),
    [{}]
  ) as Scenario[]
}

/**
 * The first turn's receipt and its prompt's acknowledgement are both lost and
 * no page reaches the window: nothing proves the store holds that turn, so the
 * window keeps it rather than risk hiding words that were never stored.
 */
const unproven = (scenario: Scenario) =>
  scenario.receipt === 'lost' && scenario.prompts === 'optimistic' && scenario.olderPages !== 'works'

const scenarioBase: Scenario = {
  receipt: 'full',
  nextReceipt: 'full',
  next: 'delegation',
  sawNext: true,
  compaction: false,
  narration: false,
  page: 'first-turn',
  olderPages: 'works',
  prompts: 'acknowledged'
}

const FIRST_TURN = ['user:Earlier prompt.', 'assistant:Earlier reply.', 'user:Inspect this.']

describe('report order after a transcript refresh', () => {
  it.each([
    { withPrefix: true, replyId: 'assistant-stream' },
    { withPrefix: false, replyId: 'assistant-stream' },
    { withPrefix: true, replyId: 'assistant' },
    { withPrefix: false, replyId: 'assistant' }
  ])('keeps a repeated unpersisted answer with prefix=$withPrefix and id=$replyId', ({ withPrefix, replyId }) => {
    const q1 = optimisticPrompt('user-q1', 'Q1', 10)

    const first: ChatMessage = {
      id: `${replyId}-1`,
      role: 'assistant',
      pending: false,
      durableComplete: false,
      parts: [{ type: 'text', text: 'Done.' }]
    }

    const q2 = optimisticPrompt('user-next', 'Q2')
    const second: ChatMessage = { ...first, id: `${replyId}-2` }
    const stored: ChatMessage = { ...first, id: '11-0-assistant', rowId: 11, durableComplete: true }

    const page: ChatMessage[] = [
      ...(withPrefix
        ? [{ ...first, id: '1-0-assistant', rowId: 1, parts: [{ type: 'text' as const, text: 'Earlier.' }] }]
        : []),
      q1,
      stored
    ]

    const refresh = (window: ChatMessage[]) =>
      preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(page, window), window)

    const window = [q1, first, q2, second]

    if (withPrefix && replyId === 'assistant-stream') {
      // This exercises the overlap merge. Without a prefix the graft's
      // historical anchor=0 contract returns the page by reference, and the
      // following preserve pass is responsible for the unpersisted tail.
      expect(graftRefreshedTailOntoBackfill(page, window)).toEqual([...page, q2, second])
    }

    const result = refresh(window)

    expect(result).toEqual([...page, q2, second])
    expect(refresh(result)).toEqual(result)
  })

  it('matches a re-stored report past any number of unstored narration bubbles', () => {
    const old: ChatMessage = {
      id: '10-0-assistant',
      rowId: 100,
      role: 'assistant',
      parts: [{ type: 'text', text: 'Report.' }]
    }

    const restored: ChatMessage = { ...old, id: '10-1-assistant', rowId: 300 }

    const notice: ChatMessage = {
      id: 'notice',
      rowId: 200,
      role: 'system',
      parts: [{ type: 'text', text: 'Shared notice.' }]
    }

    const narration: ChatMessage[] = [1, 2].map(n => ({
      id: `assistant-stream-n${n}`,
      role: 'assistant',
      interim: true,
      pending: false,
      parts: [{ type: 'text', text: `Progress ${n}.` }]
    }))

    const page = [restored, notice]
    const result = graftRefreshedTailOntoBackfill(page, [old, ...narration, notice])

    expect(result).toEqual([restored, ...narration, notice])
    expect(
      graftRefreshedTailOntoBackfill(page, result).filter(message => chatMessageText(message) === 'Report.')
    ).toEqual([restored])
  })

  it('rejects a truncated suffix when the server supplied the complete transcript', () => {
    const truth = ['user:Q.', 'assistant:R.', 'user:N.', 'assistant:S.']

    expect(readsLikeAReload(['assistant:S.'], truth, truth)).toBe(false)
  })
  it('keeps the first report above a background report after a compaction re-stored it', async () => {
    const { screens } = await refreshTwice({ ...scenarioBase, receipt: 'partial', compaction: true })

    const expected = [
      ...FIRST_TURN,
      'assistant:First response.',
      'system:background agent work finished',
      'assistant:Background response.'
    ]

    expect(screens).toEqual([expected, expected])
  })

  it('does not paint a reply whose receipt was lost under the turn the page carries after it', async () => {
    const { screens } = await refreshTwice({
      ...scenarioBase,
      receipt: 'lost',
      nextReceipt: 'lost',
      sawNext: false,
      page: 'next-turn',
      olderPages: 'empty'
    })

    const expected = ['system:background agent work finished', 'assistant:Background response.']

    expect(screens).toEqual([expected, expected])
  })

  it('shows narration and reply once when the page folds them into one stored bubble', async () => {
    const { screens } = await refreshTwice({
      ...scenarioBase,
      compaction: true,
      narration: true,
      prompts: 'optimistic'
    })

    const expected = [
      ...FIRST_TURN,
      'assistant:Narration one.',
      'assistant:First response.',
      'system:background agent work finished',
      'assistant:Background response.'
    ]

    expect(screens).toEqual([expected, expected])
  })

  it('keeps the first turn above the next prompt after a compaction', async () => {
    const { screens } = await refreshTwice({ ...scenarioBase, next: 'prompt', compaction: true, prompts: 'hydrated' })

    const expected = [
      ...FIRST_TURN,
      'assistant:First response.',
      'user:Next question.',
      'assistant:Background response.'
    ]

    expect(screens).toEqual([expected, expected])
  })

  it('keeps an optimistic prompt in front of the turn it opened when older pages fail', async () => {
    const { screens, truth } = await refreshTwice({
      ...scenarioBase,
      receipt: 'lost',
      page: 'next-turn',
      olderPages: 'fails',
      prompts: 'optimistic'
    })

    expect(screens).toEqual([truth, truth])
  })

  const scenarios = allScenarios()
  const groups = new Map<string, Scenario[]>()

  for (const scenario of scenarios.filter(scenario => !unproven(scenario))) {
    const key = `${scenario.receipt} receipt, next turn opened by ${scenario.next}, ${scenario.compaction ? 'compacted' : 'not compacted'}, page from ${scenario.page}`

    groups.set(key, [...(groups.get(key) ?? []), scenario])
  }

  it.each([...groups.entries()])(
    'reads like a reload, twice: %s',
    async (_name, group) => {
      const failures: string[] = []

      for (const scenario of group) {
        const { screens, truth, required } = await refreshTwice(scenario)

        if (
          !screens.every(screen => readsLikeAReload(screen, truth, required)) ||
          JSON.stringify(screens[0]) !== JSON.stringify(screens[1])
        ) {
          failures.push(
            `${describeScenario(scenario)}\n  screens: ${JSON.stringify(screens)}\n  stored:  ${JSON.stringify(truth)}`
          )
        }
      }

      expect(failures).toEqual([])
    },
    30_000
  )

  it('keeps an unproven local turn on screen, once, without inventing rows', async () => {
    const failures: string[] = []

    for (const scenario of scenarios.filter(unproven)) {
      const { screens, truth, required } = await refreshTwice(scenario)
      const screen = screens[1]
      const counts = new Map<string, number>()

      screen.forEach(atom => counts.set(atom, (counts.get(atom) ?? 0) + 1))

      const kept = screen.includes('user:Inspect this.') && screen.includes('assistant:First response.')
      const invented = screen.some(atom => !truth.includes(atom))
      const repeated = [...counts.values()].some(count => count > 1)
      const lostPageRow = required.some(atom => !screen.includes(atom))
      const moved = JSON.stringify(screens[0]) !== JSON.stringify(screens[1])

      if (!kept || invented || repeated || moved || lostPageRow) {
        failures.push(`${describeScenario(scenario)}\n  screens: ${JSON.stringify(screens)}`)
      }
    }

    expect(failures).toEqual([])
  }, 60_000)
})
