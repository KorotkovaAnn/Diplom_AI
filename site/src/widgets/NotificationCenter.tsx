import { useState } from 'react'
import { observer } from 'mobx-react-lite'
import {useRootStore} from "../stores/rootStore.tsx";


export const NotificationCenter = observer(function NotificationCenter() {
  const { ui } = useRootStore()
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="notification-center">
      <button
        type="button"
        className="icon-button"
        aria-label="Уведомления"
        onClick={() => setIsOpen((prev) => !prev)}
      >
        <span className="icon-bell" />
        {ui.notifications.length > 0 ? (
          <span className="notification-badge" aria-hidden="true">
            {ui.notifications.length}
          </span>
        ) : null}
      </button>

      {isOpen ? (
        <div className="notification-dropdown glass-panel">
          <div className="notification-dropdown-header">
            <span>Уведомления</span>
          </div>
          <ul className="notification-list">
            {ui.notifications.map((notification) => (
              <li
                key={notification.id}
                className={`notification-item notification-${notification.type}`}
              >
                <div className="notification-dot" />
                <div className="notification-content">
                  <div className="notification-title">{notification.title}</div>
                  <div className="notification-time">{notification.time}</div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
})

