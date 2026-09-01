import { apiClient } from './apiClient'

function normalizeExistingCompute(payload) {
  if (!payload) return null
  return {
    clusterId: payload.cluster_id,
    clusterName: payload.cluster_name,
    state: payload.state,
    nodeTypeId: payload.node_type_id,
    runtime: payload.runtime,
    numWorkers: payload.num_workers,
    numCores: payload.num_cores,
    memoryMb: payload.memory_mb,
    autoterminationMinutes: payload.autotermination_minutes,
    singleNode: payload.single_node,
  }
}

// The compute sizes ForecastIQ offers. Served from project presets, so
// this returns immediately without contacting Databricks.
export async function fetchComputeOptions() {
  const payload = await apiClient.get('/compute/options')
  return {
    nodeTypes: (payload.node_types ?? []).map((node) => ({
      id: node.node_type_id,
      label: node.label,
      description: node.description,
      numCores: node.num_cores,
      memoryMb: node.memory_mb,
    })),
    runtimes: (payload.runtimes ?? []).map((runtime) => ({
      key: runtime.key,
      name: runtime.name,
    })),
    defaultNodeTypeId: payload.default_node_type_id,
    defaultRuntimeKey: payload.default_runtime_key,
  }
}

// Every all-purpose cluster in the workspace the picker can offer, looked
// up separately so it never blocks the form. One workspace call regardless
// of how many clusters come back — see ComputeService.list_existing_compute.
export async function fetchExistingCompute() {
  const payload = await apiClient.get('/compute/existing')
  return {
    available: payload.available,
    message: payload.message,
    clusters: (payload.clusters ?? []).map(normalizeExistingCompute),
  }
}

// Checks the selected existing cluster can actually run the workload.
export async function validateExistingCompute(clusterId) {
  const payload = await apiClient.post('/compute/existing/validate', { cluster_id: clusterId })
  return {
    valid: payload.valid,
    message: payload.message,
    state: payload.state,
    startsOnDemand: payload.starts_on_demand,
  }
}

// Real backend validation against Databricks; an invalid result is a normal answer.
export async function validateCompute(jobCompute, { quick = false } = {}) {
  const payload = await apiClient.post('/compute/validate', {
    quick,
    job_compute: {
      node_type_id: jobCompute.nodeTypeId,
      runtime_key: jobCompute.runtimeKey,
      autoscale: jobCompute.autoscale,
      num_workers: jobCompute.numWorkers,
      min_workers: jobCompute.minWorkers,
      max_workers: jobCompute.maxWorkers,
    },
  })
  return { valid: payload.valid, message: payload.message, stage: payload.stage }
}

// The compute block the deploy request carries.
export function toComputePayload(compute) {
  if (compute.mode === 'existing_compute') {
    return { mode: 'existing_compute', cluster_id: compute.existingClusterId ?? null }
  }
  return {
    mode: 'new_job_compute',
    job_compute: {
      node_type_id: compute.nodeTypeId,
      runtime_key: compute.runtimeKey,
      autoscale: compute.autoscale,
      num_workers: compute.numWorkers,
      min_workers: compute.minWorkers,
      max_workers: compute.maxWorkers,
    },
  }
}
