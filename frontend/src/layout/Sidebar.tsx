import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/diagnosis", label: "Start Diagnosis" },
  { to: "/history", label: "Diagnostic History" },
  { to: "/resources", label: "Resources" },
  { to: "/settings", label: "Settings" }
];

export const Sidebar = () => {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-mark">C</span>
        <div className="logo-text">
          <span className="logo-title">C.H.I.R.A.N.J.E.E.V.I</span>
          <span className="logo-subtitle">AI Diagnostics</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              "nav-item" + (isActive ? " nav-item-active" : "")
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
};


