"use strict";

var _slicedToArray = function () {
  function e(e, t) {
    var i = [],
        n = !0,
        o = !1,
        s = void 0;

    try {
      for (var a, l = e[Symbol.iterator](); !(n = (a = l.next()).done) && (i.push(a.value), !t || i.length !== t); n = !0);
    } catch (e) {
      o = !0, s = e;
    } finally {
      try {
        !n && l.return && l.return();
      } finally {
        if (o) throw s;
      }
    }

    return i;
  }

  return function (t, i) {
    if (Array.isArray(t)) return t;
    if (Symbol.iterator in Object(t)) return e(t, i);
    throw new TypeError("Invalid attempt to destructure non-iterable instance");
  };
}();

Component({
  properties: {
    overlay: {
      type: Boolean,
      value: !0
    },
    overlayClose: {
      type: Boolean,
      value: !0
    },
    transition: {
      type: String,
      value: "fadeIn fadeOut"
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    show: !1,
    opened: !1,
    opening: !1,
    closed: !0,
    closing: !1,
    allowScroll: !0,
    scrollHeight: "auto",
    transitionO: {
      fadeIn: "sc-mask-fadeIn",
      fadeOut: "sc-mask-fadeOut",
      slideTopIn: "sc-dialog-slideTopIn",
      slideTopOut: "sc-dialog-slideTopOut",
      slideBottomIn: "sc-dialog-slideBottomIn",
      slideBottomOut: "sc-dialog-slideBottomOut",
      slideLeftIn: "sc-dialog-slideLeftIn",
      slideLeftOut: "sc-dialog-slideLeftOut",
      slideRightIn: "sc-dialog-slideRightIn",
      slideRightOut: "sc-dialog-slideRightOut"
    },
    tin: null,
    tout: null
  },
  ready: function () {
    var e = this.data.animation.split(" "),
        t = _slicedToArray(e, 2),
        i = t[0],
        n = t[1],
        o = this.data.transitionO;

    this.setData({
      tin: o[i],
      tout: o[n]
    });
  },
  externalClasses: ["sc-class"],
  methods: {
    _catchtouchmove: function () {
      return !0;
    },
    _close: function () {
      this.setData({
        closing: !0
      }), this.triggerEvent("close", {
        bubbles: !0
      });
    },
    _animationend: function (e) {
      var t = this,
          i = e.detail.animationName;
      "dialogFadeIn" !== i && "maskFadeIn" !== i || this._queryMultipleNodes(".sc-dialog > .sc-dialog-content").then(function (e) {
        t.setData({
          scrollHeight: e[0].height + "px",
          opening: !1,
          opened: !0,
          closed: !1
        }), t.triggerEvent("opened", {
          bubbles: !0
        });
      }), "dialogFadeOut" !== i && "maskFadeOut" !== i || (this.setData({
        closing: !1,
        closed: !0,
        opened: !1,
        show: !1
      }), this.triggerEvent("closed", {
        bubbles: !0
      }));
    },
    _queryMultipleNodes: function (e) {
      var t = this;
      return new Promise(function (i, n) {
        var o = swan.createSelectorQuery().in(t);
        o.select(e).boundingClientRect(), o.exec(function (e) {
          i(e);
        });
      });
    },
    _open: function () {
      this.setData({
        show: !0,
        opening: !0
      }), this.triggerEvent("open", {
        bubbles: !0
      });
    }
  }
});