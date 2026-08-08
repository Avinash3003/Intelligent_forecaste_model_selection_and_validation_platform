import { cn } from '../../utils/cn'

export default function Avatar({ initials = 'U', color = 'bg-brand-600', size = 'md', className }) {
  const sizes = {
    sm: 'h-7 w-7 text-xs',
    md: 'h-9 w-9 text-sm',
    lg: 'h-11 w-11 text-base',
  }
  return (
    <div
      className={cn(
        'flex items-center justify-center rounded-full font-semibold text-white shadow-sm select-none',
        color,
        sizes[size],
        className
      )}
    >
      {initials}
    </div>
  )
}
