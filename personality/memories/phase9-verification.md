# Phase 9 Verification

New features were just deployed. Walk the user through testing these 5 scenarios when they're ready. After each test, note the result and move to the next. Keep a running tab of what's been tested and what's left.

## Test Scenarios

1. **Semantic matching** -- Say something indirect with no keyword match, like "could you take a peek at the readme" or "have a look at that file". The filler clip should be task-appropriate, not a generic acknowledgment. This tests the new model2vec semantic fallback in the classifier.

2. **Trivial silence** -- Say just "yes" or "ok" or "mhm" after you (the assistant) give a statement, not a question. No filler clip should play. Just natural silence before responding. This tests trivial input detection.

3. **Question context override** -- After you ask the user a question (ending with ?), the user says "yes". A filler clip SHOULD play, because "yes" is a real answer to a question, not trivial backchannel.

4. **Smooth transition** -- Say any question or command. Listen for: filler clip, brief natural pause, then your TTS response. It should feel like one continuous speaker, not two separate audio events. This tests the new StreamComposer unified audio queue.

5. **Barge-in** -- User says something that triggers a long response. While you're speaking, they start talking. You should finish the current sentence then stop, feeling like a natural pause to listen, not a hard cut. This tests composer.pause() integration.

## Known Issues

- "nevermind" got classified as task and played "1 sec" -- should be conversational or acknowledgment, not task.

## Tracking

Keep a mental running tab during the conversation of which scenarios have been tested and the results. Summarize progress when asked or after each test. You do not need to update this file -- just track verbally in the conversation.
