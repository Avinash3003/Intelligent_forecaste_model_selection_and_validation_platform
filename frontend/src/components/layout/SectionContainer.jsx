import Card from '../ui/Card'
import SectionHeader from '../common/SectionHeader'
import { cn } from '../../utils/cn'

// Standard "card + section header" wrapper used to keep page-level
// sections visually consistent across the platform.
export default function SectionContainer({ title, subtitle, action, children, className }) {
  return (
    <Card className={cn('p-6', className)}>
      {(title || subtitle || action) && (
        <SectionHeader title={title} subtitle={subtitle} action={action} />
      )}
      {children}
    </Card>
  )
}
