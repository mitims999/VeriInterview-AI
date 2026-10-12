# VeriInterview frontend

An interviewer workspace built with React, TypeScript, Vite, and Tailwind. The interface uses warm stone surfaces, sage navigation, muted plum actions, and Avenir-based sans-serif typography.

## Run locally

Start the FastAPI backend on port 8000, then run:

```sh
npm install
npm run dev
```

Open the address printed by Vite (normally http://localhost:5173). Vite proxies `/api` and `/samples` to the backend.

## Review workflow

1. Enter the candidate name and optional role description.
2. Upload a PDF/DOCX/TXT resume plus TXT reference answers and transcript.
3. Optionally attach a video, then choose **Review interview**.

The form includes document-format examples and template downloads. **Use sample documents** fills the form with fictional candidate documents served by the backend. **Completed interviews** opens a library of three fictional completed interviews: Ananya Rao, Rohit Mehta, and Karan Verma. These bundled report snapshots work without the backend and include answers, skills, evidence, and export. They were generated from the existing `data/synthetic` fixtures using the actual scoring pipeline with TF-IDF answer matching; they are not real interview records. The backend still retains all five demo scenarios.

Reports separate integrity risk, confidence, and capability. Use Overview for signals and skills, Answers for expandable answer comparisons, and Evidence log for filtered observations. Export downloads the report as JSON.

Video analysis remains simulated by the existing backend. The UI labels this explicitly; uploaded video is not processed by real face or lip-sync models. The upload flow also lacks speech, gaze, and counterfactual evidence.

## Checks

```sh
npm run lint
npm run build
```

The layout includes responsive navigation, keyboard focus indicators, file-type feedback, reduced-motion support, and loading/error states.
