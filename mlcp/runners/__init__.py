"""Runner implementations.

`builtin` runners are dependency-free so the platform is demoable and testable
with no heavyweight ML libraries installed. Heavier runners (e.g. the optional
`sklearn_runner`) import their dependencies lazily, so a missing extra never
breaks the core."""
