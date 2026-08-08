import { Search } from 'lucide-react'
import Input from './Input'
import { cn } from '../../utils/cn'

export default function SearchBox({ placeholder = 'Search...', className, ...props }) {
  return (
    <Input
      icon={Search}
      placeholder={placeholder}
      containerClassName={cn('w-full', className)}
      {...props}
    />
  )
}
