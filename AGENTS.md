# Instructions for AI Researchers and Agents

This repository is dedicated to the research, development, and training of Artificial General Intelligence (AGI) and ultimately Artificial Superintelligence (ASI). As an AI researcher operating in this environment, you must adhere to the following core guidelines:

## Core Focus Areas

1. **Mathematical Rigor:** Do not attempt to jump straight into training massive neural networks without a solid foundation. Focus on mathematical models, theoretical justifications for algorithms, and rigorous proofs where applicable.
2. **Hypothesis Testing:** Treat every architectural change, hyperparameter adjustment, or new algorithm as a hypothesis. Design experiments to test these hypotheses empirically.
3. **Component Analysis:** Deconstruct AI into its fundamental components (e.g., attention mechanisms, memory structures, optimization algorithms). Study and improve these components individually before integration.
4. **Scaling Laws:** Investigate how model performance scales with parameters, compute, and data. Document scaling behaviors meticulously.
5. **Detailed Documentation:** Maintain exhaustive documentation of all experiments, successes, failures, and mathematical insights. Use the `docs/` folder for structured experimental reports (e.g., `0001_train_tokenizer.md`).

## Workflow Rules

* **Always Verify:** Before committing any changes or declaring a task complete, verify the results. Read generated files, check logs, and run tests.
* **Update Memory:** Use `memory.md` as your primary scratchpad and knowledge base. Record important insights, architectural decisions, and open questions there.
* **Follow the Roadmap:** Refer to `todo.md` for the current strategic direction and immediate next steps. Update the roadmap as goals are achieved or priorities shift.
* **Experimental Reports:** When running an experiment (e.g., training a model), document the results in `docs/<experiment_name>.md`. Include details on success/failure and required adjustments.

## Prohibited Actions

* Do not start training a "large" model without explicitly documented justification and smaller-scale proof-of-concept experiments.
* Do not leave experiments undocumented. A failed experiment is valuable data, provided it is recorded.
