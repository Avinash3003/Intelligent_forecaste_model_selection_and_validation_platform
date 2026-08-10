import { ArrowLeft, ArrowRight, Loader2, Rocket } from 'lucide-react'
import Button from '../../../components/ui/Button'

export default function WizardFooter({
  isFirstStep,
  isLastStep,
  deployed,
  loading,
  nextDisabled,
  onPrevious,
  onNext,
}) {
  if (deployed) return null

  return (
    <div className="mt-6 flex items-center justify-between">
      <Button variant="secondary" icon={ArrowLeft} onClick={onPrevious} disabled={isFirstStep || loading}>
        Previous
      </Button>

      {isLastStep ? (
        <Button onClick={onNext} disabled={loading}>
          {loading ? <Loader2 size={16} className="animate-spin" /> : <Rocket size={16} />}
          {loading ? 'Starting…' : 'Run Forecast'}
        </Button>
      ) : (
        <Button className="flex-row-reverse" onClick={onNext} disabled={loading || nextDisabled}>
          {loading ? <Loader2 size={16} className="animate-spin" /> : <ArrowRight size={16} />}
          {loading ? 'Please wait…' : 'Next'}
        </Button>
      )}
    </div>
  )
}
