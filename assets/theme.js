/**
 * LogCluster Theme Persistence & Initialization Script
 * 
 * Runs immediately before page render to prevent flash of wrong theme.
 * Restores saved theme from localStorage, or defaults to OS preferred color scheme.
 */

(function () {
  function getInitialTheme() {
    try {
      var stored = localStorage.getItem("logcluster-theme");
      if (stored === "dark" || stored === "light") {
        return stored;
      }
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return "dark";
      }
    } catch (e) {
      console.warn("Unable to access localStorage for theme preference:", e);
    }
    return "light";
  }

  var activeTheme = getInitialTheme();
  document.documentElement.setAttribute("data-theme", activeTheme);

  if (document.body) {
    document.body.setAttribute("data-theme", activeTheme);
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      document.body.setAttribute("data-theme", activeTheme);
    });
  }
})();
