# Direct-logit Jev-like decision model

## Purpose

This document describes a local decision model inspired by Jev's public API:

- `noul`: probability that a yes/no statement is true
- `choice`: probability distribution over supplied named criteria
- `score`: probability distribution over supplied ordered rubric levels

The implementation goal is not to reproduce Jev's unpublished architecture. The v1 goal is to consume state plus one typed question and return a calibrated typed distribution from selected next-token logits after a single prefill forward pass.

The v1 design deliberately excludes shared-state, multi-question execution. Treat that as a later performance optimization after the single-question model is correct, calibrated, and measurable.

## Non-goals

- No autoregressive output generation.
- No grammar-constrained decoding.
- No `n_predict`, token sampler, or generated label token.
- No attempt to infer or copy Jev's private model architecture.
- No custom classifier head, custom vocabulary, or model-format change in v1.

## Official Jev HTTP contract

This section records the public TypeSafe API contract as documented on September 30, 2026. It describes Jev's externally observable HTTP interface, not its internal implementation. The direct-logit design in the rest of this document is a local Jev-like implementation and intentionally starts with a smaller single-question contract.

### Endpoint and headers

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: <API_KEY>
Content-Type: application/json
```

The official examples show a redacted value in the `Authorization` header and do not document an authentication scheme such as `Bearer`. Send the header in the form supplied by TypeSafe. Keep the key outside source control and inject it through deployment secrets or environment configuration.

### Top-level request

```typescript
type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

type SystemOneRequest = {
  state: string | JsonValue[] | Record<string, JsonValue>;
  model: string;
  questions: Record<string, Question>;
};
```

| Field | Required | Contract |
| --- | --- | --- |
| `state` | Yes | String, object, or array containing the text and structured application state to evaluate. Jev is text-only; images, audio, video, and binary inputs must be converted to text first. |
| `model` | Yes | Model ID or alias. The documented stable alias is `jev-latest`; pin a version such as `jev-1.13.0` when behavior must not move with an alias. |
| `questions` | Yes | Non-empty map from caller-defined question ID to a typed question. The response uses the same IDs. Question IDs identify results but are not sent to the underlying model and do not affect inference. |

All questions in one request see the same `state`. They may mix `noul`, `choice`, and `score`, are evaluated independently, and return one answer per question. One question's answer is not hidden context for another question.

`state`, `instructions`, and supported `criteria` values may use JSON structure to preserve named fields and relationships. When a question targets a nested field, the public guidance recommends naming its dot-and-index path in backticks, for example:

```json
{
  "type": "noul",
  "instructions": "Does `ticket.messages[0].text` request a refund?"
}
```

### Question union

```typescript
type Description = string | JsonValue[] | Record<string, JsonValue>;

type NoulQuestion = {
  type: "noul";
  instructions: Description;
  criteria?: {
    true?: Description;
    false?: Description;
  };
};

type ChoiceQuestion = {
  type: "choice";
  instructions: Description;
  criteria: Record<string, Description | null>;
};

type ScoreQuestion = {
  type: "score";
  instructions: Description;
  criteria: Description[];
};

type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;
```

#### Noul request

`noul` asks one yes/no question. `criteria` is optional and can define what true and false mean.

```json
{
  "type": "noul",
  "instructions": "Does this convey urgency?",
  "criteria": {
    "true": "Explicitly time-sensitive",
    "false": "No urgency expressed"
  }
}
```

#### Choice request

`choice` selects one caller-defined named option. `criteria` is a map whose keys are the values that may be returned in `choice`; each value is a description or `null`. The public API accepts at most 255 options.

```json
{
  "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": {
    "billing": "Payments, invoicing, and refunds",
    "technical": "Bugs, outages, and integrations",
    "sales": "Pricing, upgrades, and new accounts"
  }
}
```

#### Score request

`score` rates the state against an ordered array of descriptions. It requires at least two levels and accepts at most ten. Array position is the numeric level: the first item is level `0`, the second is level `1`, and so on.

```json
{
  "type": "score",
  "instructions": "How frustrated is the customer?",
  "criteria": [
    "Calm",
    "Frustrated",
    "Very angry"
  ]
}
```

### Complete request example

```json
{
  "state": {
    "ticket": {
      "subject": "Duplicate charge",
      "message": "I was charged twice. Please refund the duplicate."
    }
  },
  "model": "jev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle `ticket`?",
      "criteria": {
        "billing": "Payments, invoicing, and refunds",
        "technical": "Bugs, outages, and integrations",
        "sales": "Pricing, upgrades, and new accounts"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer in `ticket.message`?",
      "criteria": [
        "Calm",
        "Frustrated",
        "Very angry"
      ]
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does `ticket.message` request a refund?"
    }
  }
}
```

### Top-level response

```typescript
type SystemOneResponse = {
  model: string;
  answers: Record<string, Answer>;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
};
```

| Field | Contract |
| --- | --- |
| `model` | Versioned model ID that performed the evaluation, even when the request used an alias. Log this value for reproducibility. |
| `answers` | Map keyed by the exact question IDs supplied in the request. Each answer's `type` matches its question's `type`. |
| `usage.input_tokens` | Input token count for the request. |
| `usage.output_tokens` | Output token count reported by the service. |

### Answer union

```typescript
type NoulAnswer = {
  type: "noul";
  noul: number;
};

type ChoiceAnswer = {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
};

type ScoreAnswer = {
  type: "score";
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
};

type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;
```

#### Noul answer

`noul` is the probability of yes, from `0` for no to `1` for yes. It has no separate `confidence` field.

```json
{
  "type": "noul",
  "noul": 0.95
}
```

#### Choice answer

`choice` is the highest-probability option key. `probabilities` contains every submitted option and sums to `1`. `confidence` is a service-derived value from `0` to `1` summarizing how concentrated the distribution is; it is not the winning option's probability.

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": {
    "billing": 0.88,
    "technical": 0.12,
    "sales": 0.0
  },
  "confidence": 0.81
}
```

#### Score answer

`score` is the probability-weighted numeric level and may fall between submitted levels. `legend` maps zero-based level indices, serialized as string keys, back to their descriptions. `probabilities` maps those same string indices to probabilities that sum to `1`.

```json
{
  "type": "score",
  "score": 1.05,
  "legend": {
    "0": "Calm",
    "1": "Frustrated",
    "2": "Very angry"
  },
  "probabilities": {
    "0": 0.0,
    "1": 0.95,
    "2": 0.05
  },
  "confidence": 0.92
}
```

### Complete response example

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.88,
        "technical": 0.12,
        "sales": 0.0
      },
      "confidence": 0.81
    },
    "frustration": {
      "type": "score",
      "score": 1.05,
      "legend": {
        "0": "Calm",
        "1": "Frustrated",
        "2": "Very angry"
      },
      "probabilities": {
        "0": 0.0,
        "1": 0.95,
        "2": 0.05
      },
      "confidence": 0.92
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.99
    }
  },
  "usage": {
    "input_tokens": 392,
    "output_tokens": 65
  }
}
```

### HTTP errors and retry behavior

Errors use an HTTP status plus a JSON body describing the failure.

| Status | Meaning | Client behavior |
| --- | --- | --- |
| `401 Unauthorized` | Missing or invalid API key. | Do not retry without correcting credentials. |
| `422 Unprocessable Entity` | Request validation failed, such as a missing field or malformed question. The body identifies the offending field. | Fix the request; do not retry unchanged. |
| `429 Too Many Requests` | Account exceeded a token-per-second or request-per-second limit. | Retry with exponential backoff and honor `Retry-After` when present. |
| `529 Overloaded` | TypeSafe is temporarily overloaded. | Retry with exponential backoff. |

The official SDKs retry `429` and `529` responses by default. A direct HTTP client should implement bounded exponential backoff, jitter, cancellation, and an overall retry budget.

### Published Jev 1.13 limits

These values are service configuration rather than stable schema and may change:

| Limit | Published value |
| --- | --- |
| Total request context | 64k tokens across `state` and all questions |
| Per-question context | 32k tokens for `state` plus the longest single question |
| Choice options | At most 255 |
| Score levels | 2 to 10 |
| Rate limit | 100k tokens/second and 40 requests/second |
| Input modalities | Text in string/object/array JSON form only |

The model page warns that rate limits are dynamic. Treat response status and retry headers as authoritative rather than hard-coding the published throughput values.

### Model discovery

Accounts can list accepted aliases with:

```http
GET https://api.typesafe.ai/v1/models
Authorization: <API_KEY>
```

The response contains a `models` array whose entries have `name`, `description`, and `release_date`. Versioned IDs may be accepted by `POST /v1/systemone` even when only aliases appear in this listing.

### Sources

- [TypeSafe API reference](https://docs.typesafe.ai/api)
- [TypeSafe primitives](https://docs.typesafe.ai/primitives)
- [TypeSafe state guide](https://docs.typesafe.ai/concepts/state)
- [TypeSafe confidence guide](https://docs.typesafe.ai/confidence)
- [TypeSafe model limits and aliases](https://docs.typesafe.ai/models)

## Model contract

The local v1 narrows the official multi-question HTTP contract to one typed question per inference call. For one local request, the model consumes:

```text
typed task instructions + criteria/rubric + state + answer prefix
```

It returns a typed probability distribution from a selected subset of language-model logits:

```text
prefill final prompt position -> vocabulary logits -> allowed-label logits -> probabilities
```

The answer label is never generated. The model runs one prefill pass and the application reads the final prompt position's vocabulary logits.

### Prompt layout

Put the task and criteria before the state so state token representations can attend to them in a causal backbone:

```text
<TASK type="choice">
Which team should handle this ticket?
</TASK>

<CRITERIA>
[A] Billing: payments, invoices, charges, and refunds
[B] Technical: product defects and integrations
[C] Account: login, permissions, and account settings
</CRITERIA>

<STATE>
Subject: I was charged twice.
Message: Please refund the extra payment.
</STATE>

Answer:
```

The final prompt position can attend to all preceding tokens. The LM head produces the logits for the next token as part of prefill. The application selects the allowed-label logits and does not sample or feed an answer token back into the model.

### Allowed-label logits

Let `z_t` be the LM-head logit for token ID `t` at the final prompt position. For allowed labels with token IDs `t_0..t_n`, use:

```text
label_logits = [z_t0, z_t1, ..., z_tn]
probabilities = softmax(label_logits / temperature)
answer = argmax(probabilities)
```

For v1:

- `noul` uses `Y` and `N`.
- `choice` uses `A` through `J` for two to ten supplied criteria.
- `score` uses `A` through `J` for two to ten ordered levels.

For `noul`, `p_yes` is the restricted softmax probability of `Y`. For `score`, the expected score is the probability-weighted index of the submitted ordered labels.

The labels are positions in the current request, not global semantic labels. For example, choice label `A` means the first serialized criterion for that request.

## Label-token validation

Every permitted label must be one distinct tokenizer token in the exact answer-prefix context. Do not assume that these strings have the same token IDs:

```text
A
 A
\nA
```

Before training or inference:

1. Freeze the prompt suffix, for example `Answer:`.
2. Tokenize each permitted label after that suffix.
3. Verify that every label maps to one distinct token ID.
4. Store the token IDs with the model configuration.
5. Reject a label scheme that does not meet this requirement.

The prompt schema and tokenizer must remain unchanged for a training run. Escape user-provided content so it cannot inject structural delimiters.

No vocabulary modification is required in v1.

## Dataset design

Store raw examples as structured data. Render the prompt in the training collator, not when collecting labels.

Every record needs:

- Stable example ID.
- Group ID for leakage-safe splitting, such as customer, document, conversation, account, or source document.
- Event or collection time.
- Prompt schema version.
- State available at decision time only.
- One typed question.
- Criteria or rubric when applicable.
- Raw annotation data and the final training target.
- Data provenance and label policy.

### Noul examples

```json
{
  "id": "example-001",
  "group_id": "thread-82",
  "prompt_schema_version": "decision-v1",
  "type": "noul",
  "state": {
    "message": "Please refund the duplicate charge."
  },
  "instructions": "Is the customer explicitly asking for a refund?",
  "criteria": {
    "true": "The customer directly requests a refund.",
    "false": "The customer does not directly request a refund."
  },
  "target_probability": 1.0,
  "annotations": [
    { "annotator": "reviewer-a", "target": 1 },
    { "annotator": "reviewer-b", "target": 1 }
  ]
}
```

### Choice examples

Keep a permanent semantic ID for every criterion. Never store a training target only as a presentation position.

```json
{
  "id": "example-002",
  "group_id": "thread-82",
  "prompt_schema_version": "decision-v1",
  "type": "choice",
  "state": {
    "message": "Please refund the duplicate charge."
  },
  "instructions": "Which team should handle this ticket?",
  "criteria": [
    { "id": "billing", "text": "Payments, invoices, charges, and refunds" },
    { "id": "technical", "text": "Product defects and integrations" },
    { "id": "account", "text": "Login, permissions, and account settings" }
  ],
  "target_criterion_id": "billing"
}
```

For every training presentation:

1. Randomly permute the criteria.
2. Render the permuted criteria as labels `A..`.
3. Map `target_criterion_id` to its current label.
4. Train against that label if fine-tuning.

This prevents position bias. The model must learn the criterion meaning, not the fact that the first or last item is often correct.

Include an explicit `other`, `none`, or `needs_review` criterion when the valid answer space is not exhaustive.

### Score examples

Score levels are ordered and must not be shuffled.

```json
{
  "id": "example-003",
  "group_id": "thread-82",
  "prompt_schema_version": "decision-v1",
  "type": "score",
  "state": {
    "message": "Please refund the duplicate charge."
  },
  "instructions": "How severe is this issue?",
  "criteria": [
    "Minor inconvenience",
    "Normal support issue",
    "Customer-impacting issue",
    "Immediate escalation needed"
  ],
  "target_level": 2
}
```

The returned score is the expected rubric index and can be fractional. It is not a universal score scale.

### Ambiguity and annotator disagreement

Keep individual annotations. When disagreement is meaningful, train against a soft target distribution:

```json
{
  "target_distribution": [0.0, 0.0, 0.25, 0.75]
}
```

For an allowed-label distribution:

```text
loss_soft = -sum(target_probability[i] * log(predicted_probability[i]))
```

Do not manufacture certainty by converting genuine disagreement into a single hard label without recording it.

### Required data quality

Collect:

- Clear positive and negative cases.
- Boundary cases between adjacent score levels.
- Plausible but wrong choice criteria.
- Insufficient-evidence cases.
- Conflicting evidence cases.
- Counterfactual pairs where one relevant fact changes the answer.
- Inputs matching production length, noise, and missing-field patterns.

Do not include fields created after the decision, such as final resolution, assigned team, or later status, in the state. They are label sources, not model inputs.

## Splits and calibration

Split by the real leakage unit, not by independent rows:

- Keep all rows from one state/document/conversation/account in the same partition.
- Keep near duplicates and document chunks in the same partition.
- Prefer a later time window for the final test partition when production data changes over time.

Use four partitions:

| Partition | Purpose |
| --- | --- |
| Train | Fit model weights |
| Validation | Early stopping and hyperparameter selection |
| Calibration | Fit probability calibration only |
| Test | Final frozen evaluation only |

Do not use calibration or test examples for model selection.

Fit temperature scaling separately by task type after the zero-shot baseline or after fine-tuning:

```text
noul: one binary calibration temperature or calibrator
choice: categorical calibration temperature
score: categorical calibration temperature
```

For score tasks, preserve the full level distribution. Do not calibrate only the expected scalar score.

## Fine-tuning

Fine-tuning is optional. Start with the pretrained model and direct allowed-label logits. The zero-shot baseline establishes whether the prompt, task definition, and labels are viable before model weights change.

When fine-tuning is needed, train the normal causal language-model objective using the same prompt layout and exactly one label token as the target:

```text
<STATE>
...
</STATE>
<QUESTION>
...
</QUESTION>
<CRITERIA>
[A] ...
[B] ...
</CRITERIA>
Answer: B
```

Mask loss for all prompt tokens and train on the one answer-label token. A custom restricted-label cross-entropy objective is optional, but is not required for the initial SFT baseline.

For a score task, preserve ordered labels and evaluate ordinal metrics. Do not shuffle score levels. If fine-tuning does not meet quality or calibration requirements, a later custom decision-head model is an architectural follow-up, not part of v1.

## Inference engine

### Single-question v1

```text
1. Validate request type and criterion count.
2. Render the frozen prompt schema.
3. Tokenize the prompt.
4. Run one prefill forward pass.
5. Read vocabulary logits at the final prompt position.
6. Gather logits for the configured allowed-label token IDs.
7. Apply the calibrated restricted softmax.
8. Return the typed distribution and derived result.
```

No decode phase is allowed:

- Do not append a generated token.
- Do not invoke a sampler.
- Do not use grammar to constrain output.
- Do not parse generated text into a decision.

The inference engine needs direct access to the full final-position vocabulary logits. An endpoint that returns only top-logprob candidates is insufficient if an allowed label is not in that top set.

### Output contract

Return both the selected result and the full distribution:

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": {
    "billing": 0.84,
    "technical": 0.06,
    "account": 0.10
  }
}
```

Map output slots back to the original criterion IDs after inference. Never expose a training-time shuffled slot as a semantic answer.

For `score`, return:

- Ordered level probabilities.
- The submitted rubric.
- Expected score.
- Optional nearest level.

For `noul`, return the probability of true/yes. Treat thresholds as application policy, not as model architecture.

### Runtime integration

The v1 model uses the existing LM head and needs no model-format extension. The inference path must expose final-position vocabulary logits before sampling, gather configured label IDs, and return the restricted distribution.

## Evaluation

Measure more than accuracy.

### Noul

- Binary cross-entropy / negative log-likelihood.
- Brier score.
- AUROC and precision-recall metrics when relevant.
- Reliability diagram and expected calibration error.
- Error rate above each production threshold.

### Choice

- Accuracy and macro-F1.
- Negative log-likelihood.
- Brier score.
- Expected calibration error.
- Top-2 accuracy where operationally useful.
- Permutation consistency: shuffle criteria at evaluation time, map probabilities back to semantic IDs, and measure distribution changes.

### Score

- Negative log-likelihood of the level distribution.
- Brier score and expected calibration error.
- Mean absolute error of the expected score.
- Ordinal agreement metrics.
- Error near level boundaries.

For every task, report performance on the natural production distribution and on relevant slices. Do not claim calibration from training metrics.

## Deferred optimization: multi-question shared state

The public Jev API supports multiple typed questions over one shared state. Jev's internal architecture is not public. An independent reconstruction suggests shared-state prefill plus per-question readouts, but this is not a confirmed implementation detail.

Do not block v1 on this problem.

After single-question inference is validated, consider:

1. Batched independent question prompts. This is the simplest correct baseline, but repeats the state.
2. State-prefix KV reuse plus a batched question suffix pass. This computes state once and has zero decode steps, but uses two forward invocations.
3. A packed single-pass shared-prefix attention mask. State is visible to every question block, while blocks cannot read each other. This is the literal one-prefill solution, but requires custom mask and efficient attention-kernel support.

For the packed-mask design:

```text
state tokens:
    attend within the state prefix

question block i:
    attends to the state prefix
    attends within block i
    does not attend to other question blocks
```

Each block ends with an answer prefix. Read allowed-label logits at the final position of each block.

## Implementation milestones

1. Define the fixed prompt schema, answer prefix, allowed labels, and JSONL dataset schema.
2. Verify single-token IDs for `Y`, `N`, and `A..J` in the exact answer-prefix context.
3. Build a single-question zero-shot `noul` baseline from direct label logits.
4. Add `choice` with two to ten criteria and evaluation-time/training-time permutation.
5. Add `score` with ordered rubrics.
6. Add group/time-safe splitting, calibration, and the evaluation suite.
7. Fine-tune only if the zero-shot or few-shot baseline misses quality, robustness, calibration, or latency targets.
8. Only then design shared-state multi-question execution or a custom decision-head architecture.
