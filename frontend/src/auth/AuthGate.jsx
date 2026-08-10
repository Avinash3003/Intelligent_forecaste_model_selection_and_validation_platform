import { AlertTriangle, BadgeCheck, GitBranch, ShieldCheck } from 'lucide-react'
import { AUTH_STATUS, useAuth } from './AuthProvider'
import ForecastArtwork from './ForecastArtwork'
import MicrosoftIcon from './MicrosoftIcon'
import Loader from '../components/ui/Loader'
import SigmoidMark from '../components/brand/SigmoidMark'
import { projectInfo } from '../data/appConfig'

/**
 * Nothing renders until the user is signed in.
 *
 * Wraps the whole router rather than each route: an unauthenticated
 * browser should not be able to see the application's shell, navigation or
 * page structure at all.
 *
 * There is deliberately no e-mail field, no password field and no sign-up
 * path. Access to ForecastIQ is provisioned in Microsoft Entra ID and
 * carried by an app role, so the only way in is the organisation's own
 * identity provider — a local account form would be an unusable dead end
 * at best and a credential-phishing surface at worst.
 */
export default function AuthGate({ children }) {
  const { status, error, login } = useAuth()

  if (status === AUTH_STATUS.AUTHENTICATED) return children

  return (
    <Shell>
      {status === AUTH_STATUS.LOADING && (
        <div className="py-4">
          <Loader label="Signing you in…" />
        </div>
      )}

      {status === AUTH_STATUS.ERROR && (
        <>
          <div className="flex items-start gap-2.5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-left text-sm text-amber-800 dark:border-amber-800/60 dark:bg-amber-900/20 dark:text-amber-300">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
          <SignInButton onClick={login} label="Try again" />
        </>
      )}

      {status === AUTH_STATUS.UNAUTHENTICATED && <SignInButton onClick={login} />}
    </Shell>
  )
}

/**
 * The single primary call to action.
 *
 * Styled to Microsoft's own sign-in button guidance — white surface, grey
 * border, dark neutral label, unmodified four-square mark — because users
 * are meant to recognise it on sight. Only the radius, focus ring and
 * motion are tuned to sit beside the rest of ForecastIQ.
 */
function SignInButton({ onClick, label = 'Sign in with Microsoft' }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex w-full items-center justify-center gap-3 rounded-xl border border-slate-300 bg-white px-5 py-3.5 text-[15px] font-semibold text-slate-700 shadow-sm transition-all duration-200 hover:border-slate-400 hover:bg-slate-50 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 active:scale-[0.99] dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:hover:border-slate-500 dark:hover:bg-slate-700 dark:focus-visible:ring-offset-slate-900"
    >
      <MicrosoftIcon size={19} />
      {label}
    </button>
  )
}

function Shell({ children }) {
  return (
    <div className="grid min-h-screen lg:grid-cols-[1.05fr_1fr]">
      {/* ---------------------------------------------------------------
          Brand panel. Always dark, regardless of theme: the sign-in screen
          renders outside MainLayout and so never receives the app's theme
          class, and a panel that commits to one treatment looks deliberate
          where one that half-responds looks broken.
          Hidden below `lg` — on a phone it would push the actual sign-in
          control below the fold.
      --------------------------------------------------------------- */}
      <section className="relative hidden overflow-hidden bg-gradient-to-br from-brand-800 via-brand-700 to-indigo-900 lg:flex lg:flex-col lg:p-12">
        <ForecastArtwork />

        {/* Scrim. The artwork bleeds in from the upper right; this keeps a
            dark, even field under the copy on the lower left so text
            contrast never depends on where the animation happens to be. */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-brand-900 via-brand-900/80 to-transparent" />

        <div className="relative flex items-center gap-3.5">
          {/* White badge, not the translucent brand-tinted one used
              elsewhere: the mark's red/black only reads correctly against
              a neutral ground. Hugs the mark's real (wide) aspect ratio
              rather than forcing it into a square. */}
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white px-2 shadow-lg">
            <SigmoidMark size={54} />
          </div>
          <div>
            <p className="text-xl font-bold leading-tight tracking-tight text-white">{projectInfo.name}</p>
            <p className="text-sm text-indigo-200/80">Enterprise AutoML</p>
          </div>
        </div>

        {/* Bottom-anchored: the artwork occupies the upper field, so
            floating this block in the vertical centre would leave a dead
            band beneath it on tall viewports. */}
        <div className="relative mt-auto max-w-md pt-16">
          <h2 className="text-[28px] font-bold leading-tight tracking-tight text-white">
            Forecasts your business
            <br />
            can actually defend.
          </h2>
          <p className="mt-3 text-sm leading-relaxed text-indigo-200/90">
            Every candidate model is backtested, screened for forecast reliability, drift-gated and
            explained — then the winner is chosen on the evidence, not on lowest error alone.
          </p>

          <ul className="mt-7 space-y-3">
            {[
              { icon: GitBranch, text: 'Prophet, ARIMA, XGBoost, LightGBM and TFT, evaluated per business key' },
              { icon: BadgeCheck, text: 'Ranked on accuracy, stability and explainability — then drift-gated' },
              { icon: ShieldCheck, text: 'Every run tracked in MLflow with a complete audit trail' },
            ].map(({ icon: Icon, text }) => (
              <li key={text} className="flex items-start gap-2.5 text-[13px] leading-snug text-indigo-100/90">
                <Icon size={15} className="mt-0.5 shrink-0 text-indigo-300" strokeWidth={2} />
                {text}
              </li>
            ))}
          </ul>
        </div>

        <p className="relative mt-10 text-[11px] text-indigo-300/60">
          © {new Date().getFullYear()} {projectInfo.name} · built by Sigmoid
        </p>
      </section>

      {/* ---------------------------------------------------------------
          Sign-in panel.
      --------------------------------------------------------------- */}
      <section className="flex items-center justify-center bg-slate-50 px-6 py-12 dark:bg-surface">
        <div className="w-full max-w-[22rem]">
          {/* Logo repeats here for the sub-`lg` viewport, where the brand
              panel is not rendered at all. */}
          <div className="mb-8 flex items-center gap-3.5 lg:hidden">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-200 bg-white px-2 shadow-sm dark:border-slate-700">
              <SigmoidMark size={54} />
            </div>
            <div>
              <p className="text-xl font-bold leading-tight tracking-tight text-slate-800 dark:text-slate-100">
                {projectInfo.name}
              </p>
              <p className="text-sm text-slate-400">Enterprise AutoML</p>
            </div>
          </div>

          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-50">
            Sign in to {projectInfo.name}
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-slate-500 dark:text-slate-400">
            Use your organization Microsoft account to securely access {projectInfo.name}.
          </p>

          <div className="mt-7 space-y-3">{children}</div>

          <div className="mt-7 flex items-start gap-2 border-t border-slate-200 pt-5 dark:border-slate-800">
            <ShieldCheck size={14} className="mt-0.5 shrink-0 text-slate-400" strokeWidth={2} />
            <div className="text-[12px] leading-relaxed text-slate-500 dark:text-slate-400">
              <p className="font-medium text-slate-600 dark:text-slate-300">
                Access is managed by your organization.
              </p>
              <p className="mt-0.5 text-slate-400 dark:text-slate-500">
                Your ForecastIQ role is assigned in Microsoft Entra ID. Contact your administrator if
                you need access.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
