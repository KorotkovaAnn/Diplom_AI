import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { observer } from 'mobx-react-lite'
import { Box, Button } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import MenuIcon from '@mui/icons-material/Menu'

const navItems: { to: string; label: string }[] = [
  { to: '/', label: 'Главная' },
  { to: '/dashboards', label: 'Дашборды' },
  { to: '/indicators', label: 'Показатели' },
  { to: '/forecasts', label: 'Прогнозы' },
  { to: '/data-upload', label: 'Загрузка данных' },
]

export const Header = observer(function Header() {
  const navigate = useNavigate()
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false)

  function closeMobileNav() {
    setIsMobileNavOpen(false)
  }

  function handleLogoClick() {
    closeMobileNav()
    navigate('/')
  }

  return (
    <Box component="header" className={`app-header glass-panel${isMobileNavOpen ? ' app-header-open' : ''}`}>
      <Box className="app-header-inner">
        <Box className="app-header-left">
          <Button
            type="button"
            className="logo-area"
            onClick={handleLogoClick}
            color="inherit"
            disableRipple
            sx={{ p: 0, minWidth: 'auto' }}
          >
            <Box className="logo-mark">
              <span className="logo-bar logo-bar-1" />
              <span className="logo-bar logo-bar-2" />
              <span className="logo-bar logo-bar-3" />
            </Box>
            <Box className="logo-text">
              <span className="logo-title">REGION AI FORECAST</span>
              <span className="logo-subtitle">B2G платформа мониторинга</span>
            </Box>
          </Button>

          <Box component="nav" className="app-nav">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `app-nav-link${isActive ? ' app-nav-link-active' : ''}`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </Box>
        </Box>

        <Box className="app-header-right">
          <button
            type="button"
            className="mobile-nav-toggle"
            aria-label={isMobileNavOpen ? 'Закрыть меню' : 'Открыть меню'}
            aria-expanded={isMobileNavOpen}
            onClick={() => setIsMobileNavOpen((current) => !current)}
          >
            {isMobileNavOpen ? <CloseIcon fontSize="small" /> : <MenuIcon fontSize="small" />}
          </button>
        </Box>
      </Box>

      <Box component="nav" className="mobile-nav" aria-hidden={!isMobileNavOpen}>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            onClick={closeMobileNav}
            className={({ isActive }) =>
              `mobile-nav-link${isActive ? ' mobile-nav-link-active' : ''}`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </Box>
    </Box>
  )
})
