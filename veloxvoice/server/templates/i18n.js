/* Minimal markdown-table i18n shared by the music client and staff views.
 *
 * Language precedence: ?lang= > localStorage > Chinese.
 * Static DOM: data-i18n, data-i18n-placeholder, data-i18n-title,
 * data-i18n-aria-label. Dynamic strings: VeloxI18n.t(key, vars).
 */
(function () {
  const SUPPORTED = ['zh', 'en'];
  const DEFAULT_LANG = 'zh';
  const state = {lang: DEFAULT_LANG, dict: {}, fallback: {}};
  let initPromise = null;

  function normalizeLang(value) {
    const lang = String(value || '').toLowerCase();
    if (lang === 'en') return 'en';
    if (lang === 'zh' || lang === 'cn' || lang === 'zh-cn') return 'zh';
    return null;
  }

  function getLang() {
    let params = null;
    try { params = new URLSearchParams(location.search); } catch (e) { /* */ }
    const fromQuery = normalizeLang(params && params.get('lang'));
    if (fromQuery) {
      try { localStorage.setItem('velox_lang', fromQuery); } catch (e) { /* */ }
      return fromQuery;
    }
    let stored = null;
    try { stored = normalizeLang(localStorage.getItem('velox_lang')); }
    catch (e) { /* */ }
    return stored || DEFAULT_LANG;
  }

  function fileFor(lang) {
    return lang === 'en' ? 'en' : 'cn';
  }

  function parseMarkdown(text) {
    const dict = {};
    for (const raw of String(text || '').split(/\r?\n/)) {
      const line = raw.trim();
      if (!line.startsWith('|')) continue;
      const cells = line.split('|').slice(1, -1).map(cell => cell.trim());
      if (cells.length < 2) continue;
      let key = cells[0].replace(/^`|`$/g, '').trim();
      if (!key || key === 'key' || /^:?-{3,}:?$/.test(key)) continue;
      dict[key] = cells.slice(1).join(' | ').trim();
    }
    return dict;
  }

  async function loadDict(lang) {
    const response = await fetch('/i18n/' + fileFor(lang) + '.md',
      {cache: 'no-store'});
    if (!response.ok) throw new Error('i18n HTTP ' + response.status);
    return parseMarkdown(await response.text());
  }

  function format(text, vars) {
    return String(text).replace(/\{(\w+)\}/g, (match, key) => {
      return vars && vars[key] !== undefined && vars[key] !== null
        ? String(vars[key]) : match;
    });
  }

  function t(key, vars) {
    const value = Object.prototype.hasOwnProperty.call(state.dict, key)
      ? state.dict[key]
      : (Object.prototype.hasOwnProperty.call(state.fallback, key)
        ? state.fallback[key] : key);
    return format(value, vars);
  }

  function apply(root) {
    const scope = root || document;
    scope.querySelectorAll('[data-i18n]').forEach(el => {
      let params = null;
      const rawParams = el.getAttribute('data-i18n-params');
      if (rawParams) {
        try { params = JSON.parse(rawParams); } catch (e) { params = null; }
      }
      el.textContent = t(el.getAttribute('data-i18n'), params);
    });
    scope.querySelectorAll('[data-i18n-html]').forEach(el => {
      el.innerHTML = t(el.getAttribute('data-i18n-html'));
    });
    scope.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      el.setAttribute('placeholder',
        t(el.getAttribute('data-i18n-placeholder')));
    });
    scope.querySelectorAll('[data-i18n-title]').forEach(el => {
      el.setAttribute('title', t(el.getAttribute('data-i18n-title')));
    });
    scope.querySelectorAll('[data-i18n-aria-label]').forEach(el => {
      el.setAttribute('aria-label',
        t(el.getAttribute('data-i18n-aria-label')));
    });
  }

  function bindSwitches() {
    document.querySelectorAll('[data-lang-button]').forEach(button => {
      const lang = normalizeLang(button.getAttribute('data-lang-button'));
      button.classList.toggle('active', lang === state.lang);
      button.setAttribute('aria-pressed', lang === state.lang ? 'true' : 'false');
      if (button.dataset.i18nBound === '1') return;
      button.dataset.i18nBound = '1';
      button.addEventListener('click', () => setLang(lang));
    });
  }

  function setLang(lang) {
    const next = normalizeLang(lang) || DEFAULT_LANG;
    try { localStorage.setItem('velox_lang', next); } catch (e) { /* */ }
    const url = new URL(location.href);
    url.searchParams.set('lang', next);
    location.href = url.toString();
  }

  function init() {
    if (initPromise) return initPromise;
    initPromise = (async () => {
      if (document.readyState === 'loading') {
        await new Promise(resolve => {
          document.addEventListener('DOMContentLoaded', resolve, {once: true});
        });
      }
      state.lang = getLang();
      document.documentElement.lang = state.lang === 'en' ? 'en' : 'zh-CN';
      const fallbackLang = state.lang === 'en' ? 'zh' : 'en';
      const [selected, fallback] = await Promise.all([
        loadDict(state.lang).catch(() => ({})),
        loadDict(fallbackLang).catch(() => ({})),
      ]);
      state.dict = selected;
      state.fallback = fallback;
      apply(document);
      bindSwitches();
      document.title = t('app.title');
      return state.lang;
    })();
    return initPromise;
  }

  window.VeloxI18n = {
    init,
    t,
    apply,
    setLang,
    get lang() { return state.lang; },
    get dict() { return state.dict; },
  };
})();
