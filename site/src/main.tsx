import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { CssBaseline, ThemeProvider } from '@mui/material'
import { App } from './app/App'
import { RootStoreProvider } from './stores/rootStore'
import { muiTheme } from './app/muiTheme'
import './app/styles.css'

const container = document.getElementById('root')

if (container) {
  createRoot(container).render(
    <StrictMode>
      <ThemeProvider theme={muiTheme}>
        <CssBaseline />
        <RootStoreProvider>
          <App />
        </RootStoreProvider>
      </ThemeProvider>
    </StrictMode>,
  )
}
