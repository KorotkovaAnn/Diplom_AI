import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { MainLayout } from '../layouts/MainLayout'
import { HomePage } from '../pages/HomePage'
import { DashboardsPage } from '../pages/DashboardsPage'
import { ForecastsPage } from '../pages/ForecastsPage'
import { IndicatorsPage } from '../pages/IndicatorsPage'
import { DataUploadPage } from '../pages/DataUploadPage'

export function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dashboards" element={<DashboardsPage />} />
          <Route path="/indicators" element={<IndicatorsPage />} />
          <Route path="/forecasts" element={<ForecastsPage />} />
          <Route path="/data-upload" element={<DataUploadPage />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  )
}

export default App
