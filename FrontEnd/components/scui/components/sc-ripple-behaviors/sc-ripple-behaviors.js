"use strict";

var _slicedToArray = function () {
  function t(t, e) {
    var i = [],
        r = !0,
        a = !1,
        l = void 0;

    try {
      for (var s, p = t[Symbol.iterator](); !(r = (s = p.next()).done) && (i.push(s.value), !e || i.length !== e); r = !0);
    } catch (t) {
      a = !0, l = t;
    } finally {
      try {
        !r && p.return && p.return();
      } finally {
        if (a) throw l;
      }
    }

    return i;
  }

  return function (e, i) {
    if (Array.isArray(e)) return e;
    if (Symbol.iterator in Object(e)) return t(e, i);
    throw new TypeError("Invalid attempt to destructure non-iterable instance");
  };
}(),
    regexp = /(\b[0-9]{1,3}\b)/g;

module.exports = Behavior({
  behaviors: [],
  properties: {
    ripple: {
      type: Boolean,
      value: !0
    }
  },
  data: {
    rippleList: [],
    rippleId: 0,
    rippleDeleteCount: 0,
    rippleDeleteTimer: null,
    rippleColor: "#ffffff",
    btnSelector: ".sc-class"
  },
  methods: {
    _addRipple_: function (t, e) {
      var i = this;
      this.properties.disabled || this._queryMultipleNodes_(this.data.btnSelector).then(function (r) {
        var a = r[0],
            l = a.width,
            s = a.height,
            p = a.left,
            o = a.top,
            n = a.backgroundColor,
            d = void 0 === n ? "rgb(255,255,255,1)" : n,
            u = r[1],
            c = u.scrollLeft,
            h = u.scrollTop,
            f = parseInt(l),
            m = parseInt(s),
            _ = d.match(regexp),
            b = _slicedToArray(_, 4),
            g = b[0],
            y = b[1],
            v = b[2],
            L = b[3],
            I = void 0 === L ? 1 : L,
            A = f > m ? f : m,
            C = t.detail.x - (p + c) - A / 2,
            D = t.detail.y - (o + h) - A / 2;

        i.data.rippleList.push({
          rippleId: "ripple-" + i.data.rippleId++,
          width: A,
          height: A,
          left: C,
          top: D,
          backgroundColor: i._rgbIsLight_(g, y, v, I) ? "rgb(0,0,0)" : "rgb(255,255,255)",
          startAnimate: !0,
          holdAnimate: e || !1
        }), i.setData({
          rippleList: i.data.rippleList
        });
      });
    },
    _queryMultipleNodes_: function (t) {
      var e = this;
      return new Promise(function (i, r) {
        e.createSelectorQuery().select(t).fields({
          size: !0,
          rect: !0,
          computedStyle: ["backgroundColor"]
        }).selectViewport().scrollOffset().exec(function (t) {
          i(t);
        });
      });
    },
    _rippleAnimationEnd_: function () {
      function t() {
        this.data.rippleList.splice(0, this.data.rippleDeleteCount), this.setData({
          rippleList: this.data.rippleList
        }), clearTimeout(this.data.timer), this.data.timer = null, this.data.rippleDeleteCount = 0;
      }

      this.data.rippleDeleteCount++, this.data.timer && clearTimeout(this.data.timer), this.data.timer = setTimeout(t.bind(this), 300);
    },
    _longPress_: function (t) {
      this._addRipple_(t, !0);
    },
    _touchEnd_: function () {
      var t = this.data.rippleList.slice(-1)[0];
      t && t.holdAnimate && (this.data.rippleList.pop(), this.setData({
        rippleList: this.data.rippleList
      }));
    },
    _rgbIsLight_: function (t, e, i, r) {
      return .299 * parseInt(t) + .578 * parseInt(e) + .114 * parseInt(i) >= 192 * r;
    }
  }
});