# Forecast IQ — Frontend

Self-service SPA for the Intelligent Forecast Model Selection & Validation Platform.

Stack: React 18 (Vite, JSX) · React Router · Tailwind CSS · Lucide React · Framer Motion.

## Scripts

```bash
npm install    # install dependencies
npm run dev    # start dev server (http://localhost:5173)
npm run build  # production build
npm run preview
```

## Structure

- `src/components/ui` — primitive design-system components (Button, Card, Badge, Input, SearchBox, Avatar, Loader, EmptyState)
- `src/components/layout` — composite layout building blocks (SectionContainer)
- `src/components/{sidebar,header,cards,tables,common}` — feature-scoped composites
- `src/pages/*` — route-level screens
- `src/layouts/MainLayout.jsx` — persistent sidebar + header shell
- `src/routes/AppRoutes.jsx` — route table
- `src/data/appConfig.js` — static application configuration (nav, model catalog, file formats, horizon scale); every run/metric/status value in the app comes from the API, not this file
