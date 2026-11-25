# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: re-exports removed to keep imports lazy; the upstream
# ``forge.core.__init__`` eagerly imported ``WorkflowRunner`` which created
# a circular import when paired with the lazy ``forge.clients.base`` chain
# inside the vendor tree. Callers should import ``runner.WorkflowRunner``
# directly from ``..core.runner``.
