# Error: Invalid JSON payload received. Unknown name "frequency_penalty"

Date: 2025-09-06

## Summary

When running an agent with custom `ModelSettings` you may encounter this API error from the model client:

Error (trace excerpt):

```
openai.BadRequestError: Error code: 400 - [{'error': {'code': 400, 'message': 'Invalid JSON payload received. Unknown name "frequency_penalty": Cannot find field.', 'status': 'INVALID_ARGUMENT', ...}}]
```

This means the remote model endpoint rejected the request because the JSON payload contained a field it does not recognize (`frequency_penalty`). The same can occur for `presence_penalty` or other provider-specific fields.

---

## What caused this error?

- The `ModelSettings` you supplied included parameters that the underlying model API does not accept. Common offenders: `frequency_penalty`, `presence_penalty`.
- This can happen when:
  - You're using a non-OpenAI provider or a Responses-style endpoint (e.g., Google Gemini or a custom base_url) that does not support those parameters.
  - The SDK attempted to serialize the `ModelSettings` into a request format that the provider doesn't recognize.

Why it matters: sending unknown fields causes the provider to return HTTP 400 (Bad Request), aborting the call and preventing the agent from running.

---

## How to fix it (practical solutions)

Choose one of the following fixes depending on your intended provider and model:

1. Remove unsupported parameters from `ModelSettings` (quick, low-risk)

- Edit the code that creates the `ModelSettings` (likely in `practice/03_agent/runners/model_practice_runner.py`) and remove `frequency_penalty` and `presence_penalty`.

Example (before):

```py
ModelSettings(temperature=0.3, top_p=0.3, frequency_penalty=0.5, presence_penalty=0.3)
```

Example (after):

```py
ModelSettings(temperature=0.3, top_p=0.3)
```

Why: Removing unsupported fields prevents them from being serialized into the request JSON that the provider rejects.

---

2. Use a client/model that supports those parameters (if you intended to use OpenAI)

- Ensure your `Agent.model` and the underlying client are configured for the OpenAI Chat/Completions API that accepts `frequency_penalty` and `presence_penalty`.
- Check for custom `base_url` or `AsyncOpenAI` wrappers pointing to other providers (e.g., Google). If present, switch to the OpenAI client or a compatible model.

Quick check (debug):

```py
print("Model used:", focused_agent.model)
print("Model settings:", focused_agent.model_settings.__dict__)
```

Why: Some providers expose different parameter sets; aligning client + model ensures compatibility.

---

3. Map or translate settings to provider-specific fields

- If you must use a non-OpenAI provider, translate the SDK model settings into the provider's accepted fields (for example, use `max_output_tokens` instead of `max_tokens`, or omit `frequency_penalty`).
- Consult the provider docs (e.g., the provider's Responses or Chat API) for the correct parameter names.

Why: Different vendors expose different tuning knobs — translation is necessary for cross-provider portability.

---

## Debugging tips

- Wrap your `Runner.run()` call with try/except to capture the full exception and print the request details.

```py
from openai import BadRequestError

try:
    result = await Runner.run(agent, prompt_text, run_config=config)
except BadRequestError as e:
    print("API rejected request — likely unsupported model_settings fields:", e)
    raise
```

- Print `focused_agent.model` and `focused_agent.model_settings` right before the call to confirm what will be sent to the API.

- If using a custom base_url or a non-OpenAI client, confirm the provider's docs for accepted model fields.

---

## Short Explanation (for README or quick reference)

- Error: The model API returned 400 because the request JSON included unknown fields (e.g., `frequency_penalty`).
- Cause: `ModelSettings` includes provider-specific fields not supported by the configured model/client.
- Fix: Remove unsupported fields, use a compatible client/model, or map settings to the provider's accepted parameters.

---

## References

- Check the project's `model_practice_runner.py` where `focused_agent` is instantiated.
- Provider docs (OpenAI / Google Gemini / custom API) for the exact parameter names.

---

If you want, I can patch `model_practice_runner.py` to remove the unsupported fields and add defensive logging and error handling — say the word and I'll make the minimal change.
