/**
 * Decorative brand artwork for the sign-in screen.
 *
 * Depicts what the product does rather than being abstract ornament: a
 * solid observed history that continues as a dashed forward forecast
 * inside a widening confidence band — which is the actual shape of every
 * result this platform produces.
 *
 * Purely decorative, so it is `aria-hidden` and carries no axis labels,
 * legend or tooltip; it encodes no real data and no identity, so the
 * single brand hue is the whole palette. It is not a chart a user reads.
 *
 * All motion is slow, looping and disabled under `prefers-reduced-motion`.
 */
export default function ForecastArtwork() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      <style>{`
        @keyframes fiq-drift-a {
          0%, 100% { transform: translate3d(0,0,0) scale(1); }
          50%      { transform: translate3d(6%, -8%, 0) scale(1.15); }
        }
        @keyframes fiq-drift-b {
          0%, 100% { transform: translate3d(0,0,0) scale(1.1); }
          50%      { transform: translate3d(-8%, 6%, 0) scale(1); }
        }
        @keyframes fiq-float {
          0%, 100% { transform: perspective(1200px) rotateX(22deg) rotateY(-28deg) rotateZ(7deg) translateY(0); }
          50%      { transform: perspective(1200px) rotateX(22deg) rotateY(-28deg) rotateZ(7deg) translateY(-16px); }
        }
        @keyframes fiq-draw   { to { stroke-dashoffset: 0; } }
        @keyframes fiq-fade   { to { opacity: 1; } }
        @keyframes fiq-pulse  {
          0%, 100% { opacity: .35; r: 5; }
          50%      { opacity: .9;  r: 7; }
        }
        .fiq-orb-a  { animation: fiq-drift-a 19s ease-in-out infinite; }
        .fiq-orb-b  { animation: fiq-drift-b 23s ease-in-out infinite; }
        .fiq-plane  { animation: fiq-float 11s ease-in-out infinite; }
        .fiq-actual { stroke-dasharray: 620; stroke-dashoffset: 620; animation: fiq-draw 2.2s ease-out .2s forwards; }
        .fiq-fcast  { stroke-dasharray: 7 7; opacity: 0; animation: fiq-fade .9s ease-out 2.1s forwards; }
        .fiq-band   { opacity: 0; animation: fiq-fade 1s ease-out 2.3s forwards; }
        .fiq-area   { opacity: 0; animation: fiq-fade 1.4s ease-out .5s forwards; }
        .fiq-dot    { animation: fiq-pulse 3.2s ease-in-out infinite; }

        @media (prefers-reduced-motion: reduce) {
          .fiq-orb-a, .fiq-orb-b, .fiq-plane, .fiq-dot { animation: none; }
          .fiq-actual { stroke-dashoffset: 0; animation: none; }
          .fiq-fcast, .fiq-band, .fiq-area { opacity: 1; animation: none; }
          .fiq-plane { transform: perspective(1200px) rotateX(22deg) rotateY(-28deg) rotateZ(7deg); }
        }
      `}</style>

      {/* Ambient depth — two slow-drifting light sources behind the plane. */}
      <div className="fiq-orb-a absolute -left-24 -top-28 h-[26rem] w-[26rem] rounded-full bg-indigo-400/25 blur-3xl" />
      <div className="fiq-orb-b absolute -bottom-32 -right-16 h-[30rem] w-[30rem] rounded-full bg-brand-500/25 blur-3xl" />

      {/* Faint grid, to read as a product surface rather than a poster. */}
      <div
        className="absolute inset-0 opacity-[0.07]"
        style={{
          backgroundImage:
            'linear-gradient(to right, #fff 1px, transparent 1px), linear-gradient(to bottom, #fff 1px, transparent 1px)',
          backgroundSize: '52px 52px',
        }}
      />

      {/* The tilted plane. Rotation lives on this wrapper so the SVG itself
          stays an ordinary flat drawing that is easy to reason about.
          Anchored to the upper right and allowed to bleed off-canvas: the
          panel's copy occupies the lower left, and artwork running under
          body text makes both unreadable. */}
      <div className="absolute -right-[22%] -top-[6%] w-[92%]">
        <div className="fiq-plane drop-shadow-2xl">
          <svg viewBox="0 0 480 300" className="w-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="fiqArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#a5b4fc" stopOpacity="0.55" />
                <stop offset="100%" stopColor="#a5b4fc" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="fiqLine" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#c7d2fe" />
                <stop offset="100%" stopColor="#ffffff" />
              </linearGradient>
            </defs>

            {/* Recessive gridlines. */}
            {[70, 120, 170, 220].map((y) => (
              <line key={y} x1="18" y1={y} x2="462" y2={y} stroke="#ffffff" strokeOpacity="0.12" strokeWidth="1" />
            ))}
            <line x1="18" y1="262" x2="462" y2="262" stroke="#ffffff" strokeOpacity="0.28" strokeWidth="1.5" />

            {/* Observed history, filled to the baseline. */}
            <path
              className="fiq-area"
              d="M20,236 C58,228 78,196 108,190 S150,206 178,176 S224,150 258,140 L258,262 L20,262 Z"
              fill="url(#fiqArea)"
            />

            {/* Widening confidence band around the forward forecast — the
                uncertainty growing with horizon is the point of the shape. */}
            <path
              className="fiq-band"
              d="M258,140 C288,131 308,104 338,94 S392,68 458,40 L458,86 C392,110 340,132 300,158 L258,148 Z"
              fill="#c7d2fe"
              fillOpacity="0.18"
            />

            {/* Observed: solid 2px. */}
            <path
              className="fiq-actual"
              d="M20,236 C58,228 78,196 108,190 S150,206 178,176 S224,150 258,140"
              fill="none"
              stroke="url(#fiqLine)"
              strokeWidth="2.5"
              strokeLinecap="round"
            />

            {/* Forecast: dashed, same weight — a continuation, not a new series. */}
            <path
              className="fiq-fcast"
              d="M258,140 C288,131 308,104 338,94 S392,68 458,40"
              fill="none"
              stroke="#ffffff"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeOpacity="0.95"
            />

            {/* Handover point between observed and forecast. */}
            <circle className="fiq-dot" cx="258" cy="140" r="5" fill="#ffffff" />
            <circle cx="258" cy="140" r="3" fill="#4f46e5" />
          </svg>
        </div>
      </div>
    </div>
  )
}
