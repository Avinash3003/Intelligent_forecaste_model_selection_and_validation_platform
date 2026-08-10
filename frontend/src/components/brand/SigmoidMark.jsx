import sigmoidMark from '../../assets/sigmoid-mark.png'

/**
 * Sigmoid's real mark — the wave-of-dots motif cropped from the actual
 * brand asset (`src/assets/sigmoid-logo.png`, supplied by the user and
 * trimmed to just the icon portion so it fits a small square badge; the
 * full lockup with the "SIGMOID" wordmark lives alongside it for spots
 * wide enough to show text).
 *
 * Rendered at its own true colors — never recolored to the app's indigo
 * accent. A logo carries its own identity; retinting it to match whatever
 * page it sits on would misrepresent the mark, not brand it.
 */
export default function SigmoidMark({ size = 24, className }) {
  return (
    <img
      src={sigmoidMark}
      alt="Sigmoid"
      width={size}
      height={size * (160 / 449)}
      className={className}
      style={{ objectFit: 'contain' }}
    />
  )
}
