

document.addEventListener("DOMContentLoaded", () => {
  /* ==========================================================
     1. NAVBAR SCROLL EFFECT
  ========================================================== */
  const nav = document.querySelector(".custom-navbar");
  document.addEventListener("scroll", () => {
    if (!nav) return;
    if (window.scrollY > 20) nav.classList.add("navbar-scrolled");
    else nav.classList.remove("navbar-scrolled");
  });

  /* ==========================================================
     2. BASIC FADE ANIMATIONS
  ========================================================== */
  document.querySelectorAll(".animate-left, .animate-right, .animate-fade")
    .forEach(el => el.style.opacity = "1");

  /* ==========================================================
     3. NAVBAR ACTIVE INDICATOR
  ========================================================== */
  (function navbarIndicator() {
    const navBar = document.querySelector(".navbar-nav");
    const links = document.querySelectorAll(".navbar-nav .nav-link");
    if (!navBar || links.length === 0) return;

    const indicator = document.createElement("span");
    indicator.classList.add("active-indicator");
    navBar.appendChild(indicator);

    const moveIndicator = (el) => {
      const r = el.getBoundingClientRect();
      const nr = navBar.getBoundingClientRect();
      indicator.style.width = r.width + "px";
      indicator.style.left = (r.left - nr.left) + "px";
    };

    const active = document.querySelector(".active-link");
    if (active) moveIndicator(active);

    links.forEach(link => {
      link.addEventListener("mouseenter", () => moveIndicator(link));
      link.addEventListener("click", () => moveIndicator(link));
    });

    navBar.addEventListener("mouseleave", () => {
      if (active) moveIndicator(active);
    });
  })();

  /* ==========================================================
     4. MODEL TABS (appearance-only)
  ========================================================== */
  (function modelTabs() {
    const tabs = document.querySelectorAll(".model-tab");
    const panes = document.querySelectorAll(".model-pane");
    const modelInput = document.getElementById("activeModelHidden");

    if (!tabs.length || !panes.length) return;

    panes.forEach(p => p.style.display = "none");
    // show first pane if exists
    const first = document.getElementById("model1-pane");
    if (first) first.style.display = "block";

    tabs.forEach(tab => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.target;

        tabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        panes.forEach(p => p.style.display = "none");
        const pane = document.getElementById(target);
        if (pane) pane.style.display = "block";

        if (modelInput)
          modelInput.value = target.includes("model1") ? "model1" : "model2";
      });
    });
  })();

  /* ==========================================================
     5. CONTACT FORM VALIDATION
  ========================================================== */
  (function contactValidation() {
    const contactContainer = document.querySelector(".contact-container");
    if (!contactContainer) return;

    const form = document.querySelector("#contactForm");
    if (!form) return;

    form.addEventListener("submit", (e) => {
      let valid = true;
      const fields = form.querySelectorAll("input, textarea");

      fields.forEach(f => {
        if (!f.value.trim()) {
          f.style.border = "2px solid #ff4d4d";
          valid = false;
        } else {
          f.style.border = "1px solid #d6ddf0";
        }
      });

      if (!valid) {
        e.preventDefault();
        alert("Please fill all fields before submitting.");
      }
    });
  })();

  /* ==========================================================
     6. ABOUT PAGE – Scroll reveal
  ========================================================== */
  (function aboutReveal() {
    const about = document.querySelector(".about-container");
    if (!about) return;
    const elements = document.querySelectorAll(".about-section, .feature-box, .tagline");

    const reveal = () => {
      elements.forEach(el => {
        if (el.getBoundingClientRect().top < window.innerHeight - 100) {
          el.style.opacity = "1";
          el.style.transform = "translateY(0)";
        }
      });
    };

    elements.forEach(el => {
      el.style.opacity = "0";
      el.style.transform = "translateY(30px)";
      el.style.transition = "all 0.8s ease-out";
    });

    window.addEventListener("scroll", reveal);
    reveal();
  })();

  /* ==========================================================
     7. LOTTIE / LOADER – simple show/hide on submit
  ========================================================== */
  (function loaderHandler() {
    const loadingWrapper = document.getElementById("loadingWrapper");
    function showLoader(show) {
      if (!loadingWrapper) return;
      loadingWrapper.style.display = show ? "block" : "none";
    }

    // hide at start
    showLoader(false);

    document.querySelectorAll("form").forEach(form => {
      form.addEventListener("submit", () => {
        showLoader(true);
        // scroll top to show loader
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    });
  })();

  /* ==========================================================
     8. Tooltips (CSS-first, JS adds accessible keyboard support)
     - Uses .label-wrapper, .tooltip-icon and .tooltip-box from CSS
  ========================================================== */
  (function tooltipAccessibility() {
    // Add focus/blur handling for keyboard users and ensure hover still works.
    const labelWrappers = document.querySelectorAll(".label-wrapper");

    labelWrappers.forEach(wrapper => {
      const icon = wrapper.querySelector(".tooltip-icon");
      const box = wrapper.querySelector(".tooltip-box");

      if (!icon || !box) return;

      // Make icon focusable for keyboard
      icon.setAttribute("tabindex", "0");
      icon.setAttribute("role", "button");
      icon.setAttribute("aria-haspopup", "true");
      icon.setAttribute("aria-expanded", "false");

      // Show on focus
      icon.addEventListener("focus", () => {
        box.classList.add("show");
        icon.setAttribute("aria-expanded", "true");
      });
      // Hide on blur
      icon.addEventListener("blur", () => {
        box.classList.remove("show");
        icon.setAttribute("aria-expanded", "false");
      });

      // Mouseenter/mouseleave to add small delay for stability
      let enterTimer = null;
      let leaveTimer = null;

      icon.addEventListener("mouseenter", () => {
        if (leaveTimer) { clearTimeout(leaveTimer); leaveTimer = null; }
        enterTimer = setTimeout(() => box.classList.add("show"), 80);
      });

      icon.addEventListener("mouseleave", () => {
        if (enterTimer) { clearTimeout(enterTimer); enterTimer = null; }
        leaveTimer = setTimeout(() => box.classList.remove("show"), 120);
      });

      // Also allow hovering the wrapper area to keep tooltip visible
      wrapper.addEventListener("mouseenter", () => {
        if (leaveTimer) { clearTimeout(leaveTimer); leaveTimer = null; }
        enterTimer = setTimeout(() => box.classList.add("show"), 80);
      });
      wrapper.addEventListener("mouseleave", () => {
        if (enterTimer) { clearTimeout(enterTimer); enterTimer = null; }
        leaveTimer = setTimeout(() => box.classList.remove("show"), 120);
      });
    });
  })();

  console.log("QualiSense final script loaded.");
});

document.querySelectorAll('.probability-item').forEach(item => {
    let p = item.getAttribute('data-prob');
    item.style.setProperty('--prob-width', p + '%');
    item.querySelector('strong').innerText = p + "%";
});


document.addEventListener("scroll", () => {
  document.querySelectorAll(".reveal").forEach(sec => {
    if (sec.getBoundingClientRect().top < window.innerHeight - 90) {
      sec.classList.add("active");
    }
  });
});
