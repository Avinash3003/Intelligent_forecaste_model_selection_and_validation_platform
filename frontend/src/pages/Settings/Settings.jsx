import { Settings as SettingsIcon } from 'lucide-react'
import PageContainer from '../../components/common/PageContainer'
import SectionContainer from '../../components/layout/SectionContainer'
import EmptyState from '../../components/ui/EmptyState'

export default function Settings() {
  return (
    <PageContainer>
      <SectionContainer
        title="Settings"
        subtitle="Workspace, identity and platform configuration"
      >
        <EmptyState
          icon={SettingsIcon}
          title="Settings coming soon"
          description="Entra ID SSO, role-based access control and workspace preferences will be configured here."
        />
      </SectionContainer>
    </PageContainer>
  )
}
