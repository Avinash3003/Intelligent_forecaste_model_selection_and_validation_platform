import { AlertCircle, FileType2 } from 'lucide-react'
import FileDropzone from '../../../components/common/FileDropzone'
import SectionContainer from '../../../components/layout/SectionContainer'
import { supportedFileFormats } from '../../../data/appConfig'

export default function StepDatasetUpload({ file, uploading, error, onFileSelect, onRemove }) {
  return (
    <div className="space-y-5">
      <SectionContainer
        title="Upload dataset"
        subtitle="Bring any time-series dataset spanning at least 24 months of history"
      >
        <FileDropzone
          file={file}
          onFileSelect={onFileSelect}
          onRemove={onRemove}
          loading={uploading}
          accept=".csv,.xlsx,.xls"
        />
        {error && (
          <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </SectionContainer>

      <SectionContainer title="Supported file formats">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {supportedFileFormats.map((f) => (
            <div
              key={f.format}
              className="flex items-center gap-3 rounded-xl border border-slate-100 bg-slate-50/60 px-4 py-3 dark:border-slate-800 dark:bg-slate-800/40"
            >
              <FileType2 size={18} className="shrink-0 text-brand-600 dark:text-brand-400" />
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-slate-200">{f.format}</p>
                <p className="text-xs text-slate-400">{f.extension}</p>
              </div>
            </div>
          ))}
        </div>
      </SectionContainer>
    </div>
  )
}
