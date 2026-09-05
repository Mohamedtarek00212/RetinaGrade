# RetinaGrade Design System

This document is the visual source of truth for the deployment application,
project report, and presentation. The palette is inspired by the warm retinal
fundus images while retaining a clear clinical and research-oriented appearance.

## Visual direction

**Theme:** Clinical Retina

Use neutral charcoal surfaces to keep fundus images visually dominant. Retina
orange is the main brand accent, medical teal distinguishes analytical and
informational elements, and crimson is reserved for high-severity or destructive
states. Avoid gradients and interfaces dominated by orange or brown.

## Interface palette

| Token | Role | Color |
|---|---|---:|
| `background` | Main application background | `#0C0B0B` |
| `surface` | Page sections and side panels | `#171414` |
| `surface-raised` | Controls and result containers | `#242020` |
| `primary` | Primary action and brand accent | `#E86A33` |
| `primary-hover` | Primary hover and active highlight | `#FF9A56` |
| `critical` | High-risk emphasis and errors | `#B93A43` |
| `analytical` | Charts, readiness, and neutral information | `#36B5A5` |
| `text-primary` | Main text | `#F5F3F0` |
| `text-secondary` | Supporting text | `#AAA3A0` |
| `border` | Dividers and control borders | `#393231` |

## Disease grade palette

Grade colors communicate ordered severity and must not replace the written grade
or class label. Always show color together with text.

| Grade | Label | Color |
|---:|---|---:|
| `0` | No DR | `#3BB273` |
| `1` | Mild DR | `#D4B942` |
| `2` | Moderate DR | `#E8892E` |
| `3` | Severe DR | `#D9573F` |
| `4` | Proliferative DR | `#A92F48` |

## CSS tokens

```css
:root {
  --color-background: #0c0b0b;
  --color-surface: #171414;
  --color-surface-raised: #242020;
  --color-primary: #e86a33;
  --color-primary-hover: #ff9a56;
  --color-critical: #b93a43;
  --color-analytical: #36b5a5;
  --color-text-primary: #f5f3f0;
  --color-text-secondary: #aaa3a0;
  --color-border: #393231;

  --color-grade-0: #3bb273;
  --color-grade-1: #d4b942;
  --color-grade-2: #e8892e;
  --color-grade-3: #d9573f;
  --color-grade-4: #a92f48;
}
```

## Usage rules

- Display fundus images on a neutral black or charcoal background with a thin,
  restrained border.
- Use retina orange for the primary action only, such as **Analyze image**.
- Use medical teal for charts, model-ready status, and informational feedback.
- Reserve crimson for errors and grades 3-4; do not use it for routine actions.
- Keep text and numeric labels alongside every severity color for accessibility.
- Use solid colors rather than gradients, glows, or decorative color blobs.
- Keep the application research-focused and include the medical-use disclaimer
  near prediction results.

## Accessibility

Validate final text/background combinations against WCAG contrast requirements
during frontend implementation. Do not place white text directly on the yellow
or orange grade colors; use dark text or display those colors as indicators next
to separately rendered labels.
