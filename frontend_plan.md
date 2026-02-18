# Frontend Migration Master Plan (`frontend_plan.md`)

**Goal**: Migrate the legacy `frontend` (HTML/CSS/JS) to a modern `frontendv2` (Next.js + TypeScript) with 1:1 UI fidelity, improved performance, and better maintainability.

---

## 1. Core Philosophy: "The Safe Sidecar"
- **Zero Risk**: The original `frontend` folder is **READ-ONLY**. We verify against it but never touch it.
- **Iterative**: We build `frontendv2` page-by-page. Usage of shared components (Buttons, Cards) will evolve naturally.
- **Type Safety**: strict TypeScript for all data interfaces (Patient Data, Chat Messages, Sensor Readings).

## 2. Technology Stack & Decisions
| Aspect | Choice | Justification |
| :--- | :--- | :--- |
| **Framework** | Next.js 14 (App Router) | Standard for modern React. Good routing, optimization. |
| **Language** | TypeScript | Mandatory for large scale medical apps. |
| **Styling** | Tailwind CSS + CSS Variables | Fast development, easy to match existing "Google Sans" design. |
| **Icons** | Lucide React | Clean, consistent icons that match the current "Material" look. |
| **State** | React Context (Low complexity) | No need for Redux yet. `ChatContext` and `ScreeningContext` will suffice. |
| **Markdown** | `react-markdown` | Safe, easy rendering of AI responses. |
| **Animation** | `framer-motion` | For the "breathing" circles and smooth transitions. |

---

## 3. Deep Dive: Page-by-Page Analysis

### A. Landing Page (`index.html`) -> `app/page.tsx`
*Current State*: Simple HTML with links.
*Migration Plan*:
1.  **Layout**: `app/layout.tsx` will hold the `Sidebar`.
2.  **Hero Section**: Create `<WelcomeHero />`.
3.  **Modules**: Create `<ModuleCard />` component.
    *   *Props*: `title`, `description`, `icon`, `href`.
    *   *Styling*: Hover effects, border transitions (match `common-styles.css`).

### B. Chatbot Interface (`chatbot.html`) -> `app/chatbot/page.tsx`
*Current State*: ~800 lines of mixed UI and socket logic.
*Component Breakdown*:
1.  **`<ChatLayout>`**: The main container with the sidebar.
2.  **`<MessageList>`**: Scrollable area. Needs `useEffect` to scroll to bottom on new messages.
3.  **`<MessageBubble>`**:
    *   *Props*: `role` ('user' | 'assistant'), `content` (string), `isLoading` (boolean).
    *   *Features*: Renders Markdown, handles the "Thinking..." state stylization.
4.  **`<InputArea>`**: Textarea with auto-resize.
5.  **`<CitationSidebar>`**: The slide-out panel for sources.

*Logic & Hooks*:
*   **`useChatSession()`**:
    *   Manages `messages` array.
    *   Handles `POST /api/v1/doctor/chat` and processes the **ReadableStream** (SSE).
    *   Parses "status" events (e.g., `{"type": "status", "stage": "searching_pubmed"}`) to update the UI "Typing Indicator".
    *   Parses "citation" events to populate the sidebar.

### C. Health Screening (`old_index.html`) -> `app/screening/page.tsx`
*Current State*: ~1250 lines of complex hardware interaction. This is the hardest part.
*Component Breakdown*:
1.  **`<CameraFeed>`**:
    *   *Native Implementation*: Use `<img>` tag pointing to `http://localhost:8000/.../video-feed` (simplest migration).
    *   *Future*: WebRTC (later).
2.  **`<SensorGrid>`**:
    *   Displays Cards for "Radar", "Thermal", "Camera".
    *   Status: `connected` | `disconnected` | `error`.
3.  **`<GuidanceOverlay>`**: The "Move Back", "Hold Still" text over the camera.
4.  **`<ScreeningSteps>`**: The vertical stepper (Init -> Vitals -> Analysis).

*Logic & Hooks*:
*   **`useHardwareStatus()`**: Polls `/health` or acts on WebSocket events to update sensor connection status.
*   **`useScreeningMachine()`**: A State Machine (using `useReducer`) to manage the valid transitions:
    *   `IDLE` -> `CHECKING_SENSORS` -> `COUNTDOWN` -> `RECORDING` -> `ANALYZING` -> `RESULTS`.

---

## 4. Implementation Stages (The "Doable" Chunks)

### Phase 1: Foundation (The Skeleton)
- Setup `frontendv2`.
- Configure `globals.css` with the *exact* hex codes from `common-styles.css`.
- Create `components/ui/` folder with `Button`, `Card`, `Input`.

### Phase 2: The Logic-Less UI (The Skin)
- Build `app/chatbot/page.tsx` completely static.
- Hardcode some "fake" messages to verify the bubble styles and markdown rendering.
- Verify the "Sidebar" toggle animation works (using Framer Motion).

### Phase 3: The Brain Transplant (The Logic)
- **Chatbot**: Port the `fetch` loop.
    - *Challenge*: The original uses `TextDecoder` and splits by newline. We need to replicate this exactly in the `useChatSession` hook.
- **Screening**: Port the `checkSensors()` function.
    - *Challenge*: CORS issues might arise. We need to configure `next.config.js` rewrites to proxy `/api` calls to `localhost:8000`.

### Phase 4: Polish & Parity
- **Animations**: The "breathing" circle in the screening guide.
- **Error Handling**: What happens if the backend is offline? (Original shows a "Disconnected" badge).
- **Responsive Design**: Verify it works on tablet sizes (as per CSS media queries).

---

## 5. Technical Risks & Mitigations
| Risk | Mitigation |
| :--- | :--- |
| **CORS / Proxy** | Next.js Server Actions or `rewrites` in `next.config.js` to handle backend comms transparently. |
| **Streaming Lag** | Use `suspsense` or optimistic UI updates. Ensure `TextDecoder` chunking is efficient. |
| **Legacy CSS** | We will manually port critical CSS to Tailwind Config to ensure consistency, avoiding "global css soup". |

---

## 6. Next Steps
1.  Approve this plan.
2.  Initialize `frontendv2` folder.
3.  Begin "Phase 1: Foundation".
