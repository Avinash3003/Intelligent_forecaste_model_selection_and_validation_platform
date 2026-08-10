/**
 * The official Microsoft logo — four squares, exact brand hexes.
 *
 * Reproduced at the specified proportions and colours rather than
 * approximated with an icon-font glyph: this mark appears on a sign-in
 * button, where users recognise it as a trust signal, and a recoloured or
 * restyled version reads as a phishing page. The four hexes below are
 * Microsoft's published values and must not be themed, tinted, or made to
 * inherit `currentColor`.
 */
export default function MicrosoftIcon({ size = 18 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 21 21"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      focusable="false"
      className="shrink-0"
    >
      <rect x="1" y="1" width="9" height="9" fill="#F25022" />
      <rect x="11" y="1" width="9" height="9" fill="#7FBA00" />
      <rect x="1" y="11" width="9" height="9" fill="#00A4EF" />
      <rect x="11" y="11" width="9" height="9" fill="#FFB900" />
    </svg>
  )
}
