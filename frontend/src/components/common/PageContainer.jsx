import { motion } from 'framer-motion'
import { cn } from '../../utils/cn'

export default function PageContainer({ children, className }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className={cn('mx-auto w-full max-w-7xl px-6 py-8 lg:px-10', className)}
    >
      {children}
    </motion.div>
  )
}
