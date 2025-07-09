# Architecture Diagrams

This directory contains PlantUML diagrams for the Autonomous Tool Discovery System architecture. These diagrams are referenced in Chapter 4 of the dissertation.

## Diagrams Overview

1. **system_overview.puml** - High-level system architecture showing all layers and external connections
2. **component_architecture.puml** - Detailed component breakdown with interfaces and interactions
3. **data_flow.puml** - Complete data flow from user query to learning update
4. **sequence_diagrams.puml** - Key interaction sequences including:
   - Query Processing Sequence
   - Tool Discovery Sequence
   - Learning Update Sequence
5. **deployment_architecture.puml** - Production deployment and container architecture
6. **learning_architecture_classes.puml** - Q-learning system class diagram
7. **learning_architecture_flow.puml** - Q-learning process flow diagram

## Rendering the Diagrams

### Online Rendering
You can render these diagrams using the [PlantUML Online Server](http://www.plantuml.com/plantuml/):
1. Copy the content of any `.puml` file
2. Paste it into the online editor
3. The diagram will be automatically rendered

### Local Rendering with PlantUML

1. Install PlantUML:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install plantuml
   
   # On macOS with Homebrew
   brew install plantuml
   ```

2. Generate PNG images:
   ```bash
   plantuml -tpng *.puml
   ```

3. Generate SVG images (recommended for papers):
   ```bash
   plantuml -tsvg *.puml
   ```

### VS Code Extension
If using VS Code, install the "PlantUML" extension by jebbs for live preview:
1. Install the extension from VS Code marketplace
2. Open any `.puml` file
3. Press `Alt+D` to preview the diagram

## Integration with LaTeX

For dissertation documents, you can include the generated images:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/system_overview.png}
    \caption{High-Level System Architecture}
    \label{fig:system-overview}
\end{figure}
```

## Diagram Conventions

- **Blue boxes**: Core system components
- **Red elements**: External systems or databases
- **Green elements**: Cloud services or distributed components
- **Arrows**: Data flow or dependencies
- **Notes**: Additional context or important details

## Troubleshooting

If a diagram fails to render:

1. **Check PlantUML version**: Some syntax may require newer versions
2. **Validate syntax**: Use the PlantUML online server to test
3. **Common issues**:
   - Avoid complex component syntax with methods inside `{}`
   - Use standard arrow notation (`-->`) instead of directional (`-up-`)
   - Keep one diagram per file
   - Ensure proper closing tags (`@enduml`)

## Changes Made

- **component_architecture.puml**: Simplified to use notes instead of methods inside components
- **learning_architecture.puml**: Split into two files (classes and flow) for better compatibility