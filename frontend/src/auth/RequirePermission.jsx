import { ShieldOff } from 'lucide-react'
import { useAuth } from './AuthProvider'
import PageContainer from '../components/common/PageContainer'
import Card from '../components/ui/Card'

/**
 * Route guard: renders `children` only if the signed-in user holds
 * `permission`, and otherwise explains why rather than redirecting.
 *
 * A silent redirect to the dashboard reads as a bug; naming the missing
 * access tells an analyst exactly what to ask an administrator for.
 *
 * This is presentation only. Every guarded operation is enforced again
 * server-side, so bypassing this component gains nothing but a 403.
 */
export default function RequirePermission({ permission, children }) {
  const { can } = useAuth()
  if (can(permission)) return children

  return (
    <PageContainer>
      <Card className="flex flex-col items-center gap-2 px-6 py-16 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-slate-100 text-slate-400 dark:bg-slate-800">
          <ShieldOff size={22} strokeWidth={1.75} />
        </div>
        <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100">
          You don&apos;t have access to this page
        </h2>
        <p className="max-w-sm text-sm text-slate-400">
          Your ForecastIQ role doesn&apos;t include this area. Ask an administrator if you need it.
        </p>
      </Card>
    </PageContainer>
  )
}
