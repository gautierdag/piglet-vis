# Original Piglet Paper Architecture (NLU Task)

## During Pre-Training Phase

```mermaid
graph TD;
    B{Action Encoder} --> D{Action Apply};
    C{Object Encoder} --> D;
    C -- Original Object Representations --> E
    D -- Fused Representation --> E{Object Decoder};
    F[Symbolic Action Representation]:::inputclass --> B
    G[Symbolic Object Representation Pre-Action]:::inputclass --> C
    E --> I(Symbolic Object Representations Post-Action)
    classDef inputclass fill:green;
```

## During Fine-Tuning Phase (Annotated Dataset)

```mermaid
graph TD;
    B{Action LM Encoder} --> D{Action Apply};
    C{Object Encoder} --> D;
    C -- Original Object Representations --> E
    D -- Fused Representation --> E{Object Decoder};
    F[Action Text]:::inputclass --> B
    G[Symbolic Object Representation Pre-Action]:::inputclass --> C
    E --> I(Symbolic Object Representations Post-Action)
    classDef inputclass fill:green;
```

# Our Approach (NLU + Scene Comprehension)


## During Pre-Training Phase

```mermaid
graph TD;
    B{Action Encoder} --> D{Action Apply};
    A{Image Model} --> D{Action Apply};
    A --> G[Symbolic Object Representation Pre-Action]:::inputclass
    G[Symbolic Object Representations]:::inputclass -.loss.-> A
    F[Symbolic Action Representation]:::inputclass --> B
    K[Image Before Action]:::inputclass --> A
    L[Image After Action]:::inputclass --> A
    D --> E{Object Decoder};
    A --> E
    E --> I(Symbolic Object Representation Post-Action)
    classDef inputclass fill:green;
```

## During Fine-Tuning Phase

```mermaid
graph TD;
    B{Action LM Encoder} --> D{Action Apply};
    A{Image Model} --> D{Action Apply};
    A --> G[Symbolic Object Representation Pre-Action]:::inputclass
    G[Symbolic Object Representations]:::inputclass -.loss.-> A
    F[Action Text]:::inputclass --> B
    K[Image Before Action]:::inputclass --> A
    L[Image After Action]:::inputclass --> A
    D --> E{Object Decoder};
    A --> E
    E --> I(Symbolic Object Representation Post-Action)
    classDef inputclass fill:green;
```