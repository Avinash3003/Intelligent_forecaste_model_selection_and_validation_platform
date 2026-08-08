import { cn } from '../../utils/cn'

export default function Card({ children, className, hover = false, ...props }) {
  return (
    <div
      className={cn(
        'glass-panel rounded-2xl shadow-glass dark:shadow-glass-dark transition-all duration-300',
        hover && 'hover:-translate-y-0.5 hover:shadow-lg hover:border-brand-200 dark:hover:border-brand-800',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}
