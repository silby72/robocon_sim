// Register the offline cache -- where the browser allows one at all.
//
// On file:// navigator.serviceWorker does not exist: the API is restricted to
// secure origins and the file scheme is not one. Touching it there throws, so
// the guard is not defensive tidiness, it is the normal path for the way these
// pages are meant to be opened. Offline support is a property of hosting them,
// not of the pages themselves.
//
// Exposes window.OMNI_SW_STATE for the pages to report honestly rather than
// claiming an offline capability they do not have.
(function () {
  "use strict";
  if (location.protocol === "file:" || !("serviceWorker" in navigator)) {
    window.OMNI_SW_STATE = location.protocol === "file:"
      ? "file:// — Service Worker 不可（ブラウザの HTTP キャッシュのみ）"
      : "この環境では Service Worker が使えません";
    return;
  }
  window.OMNI_SW_STATE = "登録中…";
  navigator.serviceWorker.register("sw.js")
    .then(function (reg) {
      window.OMNI_SW_STATE = "オフライン対応 有効"
        + (reg.active ? "（キャッシュ済み）" : "（初回取得後に有効）");
      document.dispatchEvent(new CustomEvent("omni-sw"));
    })
    .catch(function (err) {
      window.OMNI_SW_STATE = "登録に失敗: " + (err.message || err);
      document.dispatchEvent(new CustomEvent("omni-sw"));
    });
})();
