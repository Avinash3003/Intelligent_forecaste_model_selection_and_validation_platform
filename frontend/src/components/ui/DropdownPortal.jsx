import { cloneElement, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

// Space kept clear from the viewport edge, and the floor below which a
// panel switches from "shrink to fit" to "flip to the other side" — a
// panel squeezed under 160px tall is no longer usefully scrollable.
const VIEWPORT_MARGIN = 8
const MIN_USABLE_HEIGHT = 160
// An absolute cap on the panel, independent of how much room the viewport
// happens to offer. Without it the list simply grows with the number of
// options — a run selector opened near the top of the page had ~700px of
// space below it and used all of it, so the dropdown got taller with every
// run submitted. Roughly eight rows, then it scrolls.
const MAX_PANEL_HEIGHT = 320

// Renders `children` into document.body, positioned under (or, when there
// is not enough room below, above) `anchorRef`.
//
// Two independent problems are solved here, both standard for any portaled
// popover:
//
// 1. Stacking context escape. Dropdown panels (Select/MultiSelect) are
//    anchored inside Cards that use `backdrop-blur`, which creates a new
//    CSS stacking context — a later sibling Card (also backdrop-blur) then
//    paints on top of an open dropdown regardless of z-index, since
//    z-index only resolves within the same stacking context. Portaling to
//    <body> escapes every ancestor's stacking context and overflow
//    clipping.
//
// 2. Viewport collision. The panel used to always open downward with no
//    boundary check, so a field low on a long form (e.g. Fallback Model,
//    the last field on the Forecast Configuration step) could open a list
//    that ran past the bottom of the browser window — clipped, not
//    scrollable, effectively broken. This adds the two-part fix every
//    popover library uses: flip to open upward when there's more room
//    there, and cap height with internal scroll as a safety net
//    regardless of which side it opens on.
export default function DropdownPortal({ anchorRef, open, className, children }) {
  const [placement, setPlacement] = useState(null)
  const panelRef = useRef(null)

  useLayoutEffect(() => {
    if (!open || !anchorRef.current) return

    function updatePlacement() {
      const rect = anchorRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - rect.bottom - VIEWPORT_MARGIN
      const spaceAbove = rect.top - VIEWPORT_MARGIN
      // Prefer below (the natural reading direction); flip only when below
      // is genuinely cramped and above offers more room to work with.
      const openUpward = spaceBelow < MIN_USABLE_HEIGHT && spaceAbove > spaceBelow

      setPlacement({
        left: rect.left,
        width: rect.width,
        openUpward,
        top: openUpward ? undefined : rect.bottom + 6,
        bottom: openUpward ? window.innerHeight - rect.top + 6 : undefined,
        maxHeight: Math.min(
          MAX_PANEL_HEIGHT,
          Math.max(MIN_USABLE_HEIGHT, openUpward ? spaceAbove : spaceBelow),
        ),
      })
    }

    updatePlacement()
    window.addEventListener('resize', updatePlacement)
    // Capture phase so scroll on any nested scroll container is caught too
    // ('scroll' doesn't bubble, but capture listeners still see it).
    window.addEventListener('scroll', updatePlacement, true)
    return () => {
      window.removeEventListener('resize', updatePlacement)
      window.removeEventListener('scroll', updatePlacement, true)
    }
  }, [open, anchorRef])

  if (!open || !placement) return null

  // The height cap and scroll go on `children` itself (the panel with the
  // rounded border/shadow), not this wrapper — so the visible panel keeps
  // its own rounded edges and shadow, with just its option list scrolling,
  // rather than the whole panel (chrome included) being clipped as one
  // block by an invisible outer boundary.
  const panel = cloneElement(children, {
    style: {
      ...children.props.style,
      maxHeight: placement.maxHeight,
      overflowY: 'auto',
    },
  })

  return createPortal(
    <div
      ref={panelRef}
      data-dropdown-portal=""
      className={className}
      style={{
        position: 'fixed',
        top: placement.top,
        bottom: placement.bottom,
        left: placement.left,
        width: placement.width,
      }}
    >
      {panel}
    </div>,
    document.body
  )
}
