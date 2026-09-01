import { AlertCircle, CheckCircle2, Loader2, Server, ShieldCheck } from 'lucide-react'
import SectionContainer from '../../../components/layout/SectionContainer'
import Button from '../../../components/ui/Button'
import Card from '../../../components/ui/Card'
import Input from '../../../components/ui/Input'
import Loader from '../../../components/ui/Loader'
import Select from '../../../components/ui/Select'
import { cn } from '../../../utils/cn'

const MODES = [
  {
    id: 'new_job_compute',
    title: 'Create job compute',
    description: 'Provision compute for this run. Databricks releases it when the run finishes.',
  },
  {
    id: 'existing_compute',
    title: 'Use existing compute',
    description: 'Reuse the compute already available. Nothing new is created.',
  },
]

function formatMemory(memoryMb) {
  if (!memoryMb) return null
  return `${Math.round(memoryMb / 1024)} GB`
}

function ModeCard({ mode, selected, onSelect }) {
  return (
    <button
      type="button"
      onClick={() => onSelect(mode.id)}
      className={cn(
        'flex-1 rounded-xl border p-4 text-left transition-all',
        selected
          ? 'border-brand-600 bg-brand-50/60 ring-2 ring-brand-100 dark:bg-brand-900/20 dark:ring-brand-900/40'
          : 'border-slate-200 bg-white hover:border-slate-300 dark:border-slate-700 dark:bg-slate-900'
      )}
    >
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2',
            selected ? 'border-brand-600' : 'border-slate-300 dark:border-slate-600'
          )}
        >
          {selected && <span className="h-2 w-2 rounded-full bg-brand-600" />}
        </span>
        <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">{mode.title}</p>
      </div>
      <p className="mt-1.5 pl-6 text-xs text-slate-500 dark:text-slate-400">{mode.description}</p>
    </button>
  )
}

function Field({ label, hint, children }) {
  return (
    <div className="min-w-0">
      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </label>
      {children}
      {hint && <p className="mt-1 text-xs text-slate-400">{hint}</p>}
    </div>
  )
}

function ValidationBanner({ state, message, idleText, busyText }) {
  if (state === 'idle') {
    return <p className="text-xs text-slate-400">{idleText}</p>
  }
  if (state === 'validating') {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
        <Loader2 size={15} className="animate-spin" />
        {busyText}
      </div>
    )
  }

  const valid = state === 'valid'
  const Icon = valid ? CheckCircle2 : AlertCircle
  return (
    <div
      className={cn(
        'flex items-start gap-2.5 rounded-xl border px-4 py-3 text-sm',
        valid
          ? 'border-emerald-200 bg-emerald-50/70 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-900/20 dark:text-emerald-300'
          : 'border-rose-200 bg-rose-50/70 text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300'
      )}
    >
      <Icon size={16} className="mt-0.5 shrink-0" />
      <p className="min-w-0">{message}</p>
    </div>
  )
}

// One line of metadata per cluster in the dropdown, so a workspace with
// several clusters reads as "which one" rather than a wall of identical
// names — cores/memory are exactly what tells two clusters apart at a
// glance.
function clusterSublabel(cluster) {
  const capacity = cluster.numCores ? `${cluster.numCores} vCPU · ${formatMemory(cluster.memoryMb) ?? '—'}` : null
  const shape = cluster.singleNode ? 'single node' : `${cluster.numWorkers} worker(s)`
  return [capacity, shape, cluster.state].filter(Boolean).join(' · ')
}

function ExistingComputeCard({ result, loading, selectedClusterId, validation, onValidate, onSelectCluster }) {
  if (loading) {
    return <Loader label="Loading available compute…" />
  }
  if (!result?.available || !result.clusters?.length) {
    return (
      <p className="text-sm text-slate-400">
        {result?.message ?? 'No existing compute is available in this workspace.'}
      </p>
    )
  }

  const clusters = result.clusters
  const compute = clusters.find((c) => c.clusterId === selectedClusterId) ?? clusters[0]

  const clusterOptions = clusters.map((cluster) => ({
    value: cluster.clusterId,
    label: cluster.clusterName,
    sublabel: clusterSublabel(cluster),
  }))

  const rows = [
    ['Status', compute.state],
    ['Machine type', compute.nodeTypeId],
    ['Runtime', compute.runtime],
    ['Capacity', compute.numCores ? `${compute.numCores} vCPU · ${formatMemory(compute.memoryMb)}` : '—'],
    ['Workers', compute.singleNode ? 'Single node' : `${compute.numWorkers} worker(s)`],
    [
      'Auto termination',
      compute.autoterminationMinutes ? `${compute.autoterminationMinutes} min idle` : 'Not set',
    ],
  ]

  return (
    <Card className="p-5">
      <div className="mb-4 flex items-center gap-2">
        <Server size={16} className="text-brand-600" />
        <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">Existing compute</p>
        <span className="ml-auto text-xs text-slate-400">
          {clusters.length} cluster{clusters.length === 1 ? '' : 's'} available
        </span>
      </div>

      <Field label="Cluster" hint="Every all-purpose cluster in the workspace, fetched live — nothing here is fixed in configuration.">
        <Select value={compute.clusterId} onChange={onSelectCluster} options={clusterOptions} placeholder="Select a cluster" />
      </Field>

      <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 border-t border-slate-100 pt-5 dark:border-slate-800 sm:grid-cols-3">
        {rows.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</dt>
            <dd className="mt-0.5 truncate text-sm font-medium text-slate-700 dark:text-slate-200">
              {value || '—'}
            </dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-xs text-slate-400">
        This compute is reused as-is and keeps its own auto-termination policy.
      </p>

      <div className="mt-5 flex flex-col gap-3 border-t border-slate-100 pt-5 dark:border-slate-800">
        <div>
          <Button
            variant="secondary"
            icon={ShieldCheck}
            onClick={onValidate}
            loading={validation.state === 'validating'}
          >
            Validate selected compute
          </Button>
        </div>
        <ValidationBanner
          state={validation.state}
          message={validation.message}
          idleText="Check this compute is available before continuing."
          busyText="Checking the selected compute…"
        />
      </div>
    </Card>
  )
}

export default function StepComputeConfiguration({
  compute,
  options,
  existingCompute,
  existingComputeLoading,
  validation,
  existingValidation,
  onChange,
  onValidate,
  onValidateExisting,
  onSelectExistingCluster,
}) {
  const isNew = compute.mode === 'new_job_compute'
  const nodeOptions = (options?.nodeTypes ?? []).map((node) => ({
    value: node.id,
    label: node.label ?? node.id,
    sublabel: node.description,
  }))
  const runtimeOptions = (options?.runtimes ?? []).map((runtime) => ({
    value: runtime.key,
    label: runtime.name,
  }))
  const selectedNode = options?.nodeTypes?.find((node) => node.id === compute.nodeTypeId)

  const update = (patch) => onChange({ ...compute, ...patch })

  return (
    <SectionContainer
      title="Compute"
      subtitle="Choose where this forecast runs."
    >
      <div className="space-y-6">
        <div className="flex flex-col gap-3 sm:flex-row">
          {MODES.map((mode) => (
            <ModeCard
              key={mode.id}
              mode={mode}
              selected={compute.mode === mode.id}
              onSelect={(id) => update({ mode: id })}
            />
          ))}
        </div>

        {isNew ? (
          <div className="space-y-5">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="Machine type"
                hint={
                  selectedNode?.numCores
                    ? `${selectedNode.numCores} vCPU · ${formatMemory(selectedNode.memoryMb)} per machine`
                    : undefined
                }
              >
                <Select
                  value={compute.nodeTypeId}
                  onChange={(value) => update({ nodeTypeId: value })}
                  options={nodeOptions}
                  placeholder="Select a machine type"
                />
              </Field>
              <Field label="Runtime" hint="Machine learning runtimes only — the engine requires them.">
                <Select
                  value={compute.runtimeKey}
                  onChange={(value) => update({ runtimeKey: value })}
                  options={runtimeOptions}
                  placeholder="Select a runtime"
                />
              </Field>
            </div>

            <div>
              <label className="flex items-center gap-2.5">
                <input
                  type="checkbox"
                  checked={compute.autoscale}
                  onChange={(event) => update({ autoscale: event.target.checked })}
                  className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                />
                <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
                  Scale workers automatically
                </span>
              </label>

              <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                {compute.autoscale ? (
                  <>
                    <Field label="Minimum workers">
                      <Input
                        type="number"
                        min={0}
                        value={compute.minWorkers}
                        onChange={(event) => update({ minWorkers: Number(event.target.value) })}
                      />
                    </Field>
                    <Field label="Maximum workers">
                      <Input
                        type="number"
                        min={1}
                        value={compute.maxWorkers}
                        onChange={(event) => update({ maxWorkers: Number(event.target.value) })}
                      />
                    </Field>
                  </>
                ) : (
                  <Field label="Workers" hint="Zero runs everything on a single machine.">
                    <Input
                      type="number"
                      min={0}
                      value={compute.numWorkers}
                      onChange={(event) => update({ numWorkers: Number(event.target.value) })}
                    />
                  </Field>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-3 border-t border-slate-100 pt-5 dark:border-slate-800">
              <div>
                <Button
                  variant="secondary"
                  icon={ShieldCheck}
                  onClick={onValidate}
                  loading={validation.state === 'validating'}
                  disabled={!compute.nodeTypeId || !compute.runtimeKey}
                >
                  Validate configuration
                </Button>
              </div>
              <ValidationBanner
                state={validation.state}
                message={validation.message}
                idleText="Check this configuration against Databricks before continuing."
                busyText="Checking this configuration against Databricks…"
              />
            </div>
          </div>
        ) : (
          <ExistingComputeCard
            result={existingCompute}
            loading={existingComputeLoading}
            selectedClusterId={compute.existingClusterId}
            validation={existingValidation}
            onValidate={onValidateExisting}
            onSelectCluster={onSelectExistingCluster}
          />
        )}
      </div>
    </SectionContainer>
  )
}
