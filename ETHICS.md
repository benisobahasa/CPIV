# Ethics and responsible use

## Intended use
Research on explainable voice forensics, spoofing robustness and benchmark design;
defensive tooling (for example, triaging why a recording differs from a reference).

## Dual-use
Knowing which acoustic evidence reveals a manipulation can help someone hide it. Mitigations
in this project:
- The toolkit attributes causes; it does not ship optimisation code to evade detectors.
- Voice conversion is only a lightweight proxy; no voice-cloning models are included.
- Please report misuse concerns through GitHub issues marked `ethics`.

## What CPIV must not be used for
- **Medical, emotional or psychological inference.** The `natural_variation` cause means
  vocal effort, speaking rate and level only. It is not a health or emotion claim.
- **Sole evidence in legal or disciplinary decisions.** Outputs are probabilistic and come
  from a research prototype. Expert human review is required.
- Identifying or profiling individuals without their consent.

## Data
- Respect corpus licences and consent terms. Do not publish derived audio of corpora whose
  licence forbids redistribution (the manifest and features can usually be shared instead).
- Never commit private recordings to the repository.

## Honesty about claims
Report which results come from synthetic data, simulated channels or proxies. Describe
this work as "a new approach to explainable cause attribution with a controlled benchmark",
not as technology without competitors.
