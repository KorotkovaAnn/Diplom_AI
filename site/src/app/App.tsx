import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { MainLayout } from '../layouts/MainLayout'
import { HomePage } from '../pages/HomePage'
import { DashboardsPage } from '../pages/DashboardsPage'
import { ForecastsPage } from '../pages/ForecastsPage'
import { ScenarioModelingPage } from '../pages/ScenarioModelingPage'
import { AboutPage } from '../pages/AboutPage'
import { DocsPage } from '../pages/DocsPage'

export function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dashboards" element={<DashboardsPage />} />
          <Route path="/forecasts" element={<ForecastsPage />} />
          <Route path="/scenarios" element={<ScenarioModelingPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/docs" element={<DocsPage />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  )
}

export default App