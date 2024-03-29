"use strict";

var _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (t) {
  return typeof t;
} : function (t) {
  return t && "function" == typeof Symbol && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
};

!function (t, e) {
  "object" == ("undefined" == typeof exports ? "undefined" : _typeof(exports)) && "undefined" != typeof module ? module.exports = e() : "function" == typeof define && define.amd ? define(e) : t.dayjs = e();
}(void 0, function () {
  var t = "millisecond",
      e = "second",
      n = "minute",
      r = "hour",
      s = "day",
      i = "week",
      a = "month",
      u = "year",
      o = /^(\d{4})-?(\d{1,2})-?(\d{0,2})(.*?(\d{1,2}):(\d{1,2}):(\d{1,2}))?.?(\d{1,3})?$/,
      c = /\[.*?\]|Y{2,4}|M{1,4}|D{1,2}|d{1,4}|H{1,2}|h{1,2}|a|A|m{1,2}|s{1,2}|Z{1,2}|SSS/g,
      h = {
    name: "en",
    weekdays: "Sunday_Monday_Tuesday_Wednesday_Thursday_Friday_Saturday".split("_"),
    months: "January_February_March_April_May_June_July_August_September_October_November_December".split("_")
  },
      d = function (t, e, n) {
    var r = String(t);
    return !r || r.length >= e ? t : "" + Array(e + 1 - r.length).join(n) + t;
  },
      f = {
    padStart: d,
    padZoneStr: function (t) {
      var e = Math.abs(t),
          n = Math.floor(e / 60),
          r = e % 60;
      return (t <= 0 ? "+" : "-") + d(n, 2, "0") + ":" + d(r, 2, "0");
    },
    monthDiff: function (t, e) {
      var n = 12 * (e.year() - t.year()) + (e.month() - t.month()),
          r = t.clone().add(n, "months"),
          s = e - r < 0,
          i = t.clone().add(n + (s ? -1 : 1), "months");
      return Number(-(n + (e - r) / (s ? r - i : i - r)));
    },
    absFloor: function (t) {
      return t < 0 ? Math.ceil(t) || 0 : Math.floor(t);
    },
    prettyUnit: function (o) {
      return {
        M: a,
        y: u,
        w: i,
        d: s,
        h: r,
        m: n,
        s: e,
        ms: t
      }[o] || String(o || "").toLowerCase().replace(/s$/, "");
    },
    isUndefined: function (t) {
      return void 0 === t;
    }
  },
      $ = "en",
      l = {};

  l[$] = h;

  var m = function (t) {
    return t instanceof b;
  },
      y = function (t, e, n) {
    var r;
    if (!t) return null;
    if ("string" == typeof t) l[t] && (r = t), e && (l[t] = e, r = t);else {
      var s = t.name;
      l[s] = t, r = s;
    }
    return n || ($ = r), r;
  },
      p = function (t, e) {
    if (m(t)) return t.clone();
    var n = e || {};
    return n.date = t, new b(n);
  },
      S = function (t, e) {
    return p(t, {
      locale: e.$L
    });
  },
      M = f;

  M.parseLocale = y, M.isDayjs = m, M.wrapper = S;

  var b = function () {
    function h(t) {
      this.parse(t);
    }

    var d = h.prototype;
    return d.parse = function (t) {
      var e, n;
      this.$d = null === (e = t.date) ? new Date(NaN) : M.isUndefined(e) ? new Date() : e instanceof Date ? e : "string" == typeof e && /.*[^Z]$/i.test(e) && (n = e.match(o)) ? new Date(n[1], n[2] - 1, n[3] || 1, n[5] || 0, n[6] || 0, n[7] || 0, n[8] || 0) : new Date(e), this.init(t);
    }, d.init = function (t) {
      this.$y = this.$d.getFullYear(), this.$M = this.$d.getMonth(), this.$D = this.$d.getDate(), this.$W = this.$d.getDay(), this.$H = this.$d.getHours(), this.$m = this.$d.getMinutes(), this.$s = this.$d.getSeconds(), this.$ms = this.$d.getMilliseconds(), this.$L = this.$L || y(t.locale, null, !0) || $;
    }, d.$utils = function () {
      return M;
    }, d.isValid = function () {
      return !("Invalid Date" === this.$d.toString());
    }, d.$compare = function (t) {
      return this.valueOf() - p(t).valueOf();
    }, d.isSame = function (t) {
      return 0 === this.$compare(t);
    }, d.isBefore = function (t) {
      return this.$compare(t) < 0;
    }, d.isAfter = function (t) {
      return this.$compare(t) > 0;
    }, d.year = function () {
      return this.$y;
    }, d.month = function () {
      return this.$M;
    }, d.day = function () {
      return this.$W;
    }, d.date = function () {
      return this.$D;
    }, d.hour = function () {
      return this.$H;
    }, d.minute = function () {
      return this.$m;
    }, d.second = function () {
      return this.$s;
    }, d.millisecond = function () {
      return this.$ms;
    }, d.unix = function () {
      return Math.floor(this.valueOf() / 1e3);
    }, d.valueOf = function () {
      return this.$d.getTime();
    }, d.startOf = function (t, o) {
      var c = this,
          h = !!M.isUndefined(o) || o,
          d = function (t, e) {
        var n = S(new Date(c.$y, e, t), c);
        return h ? n : n.endOf(s);
      },
          f = function (t, e) {
        return S(c.toDate()[t].apply(c.toDate(), h ? [0, 0, 0, 0].slice(e) : [23, 59, 59, 999].slice(e)), c);
      };

      switch (M.prettyUnit(t)) {
        case u:
          return h ? d(1, 0) : d(31, 11);

        case a:
          return h ? d(1, this.$M) : d(0, this.$M + 1);

        case i:
          return d(h ? this.$D - this.$W : this.$D + (6 - this.$W), this.$M);

        case s:
        case "date":
          return f("setHours", 0);

        case r:
          return f("setMinutes", 1);

        case n:
          return f("setSeconds", 2);

        case e:
          return f("setMilliseconds", 3);

        default:
          return this.clone();
      }
    }, d.endOf = function (t) {
      return this.startOf(t, !1);
    }, d.$set = function (i, o) {
      switch (M.prettyUnit(i)) {
        case s:
          this.$d.setDate(this.$D + (o - this.$W));
          break;

        case "date":
          this.$d.setDate(o);
          break;

        case a:
          this.$d.setMonth(o);
          break;

        case u:
          this.$d.setFullYear(o);
          break;

        case r:
          this.$d.setHours(o);
          break;

        case n:
          this.$d.setMinutes(o);
          break;

        case e:
          this.$d.setSeconds(o);
          break;

        case t:
          this.$d.setMilliseconds(o);
      }

      return this.init(), this;
    }, d.set = function (t, e) {
      return this.clone().$set(t, e);
    }, d.add = function (t, o) {
      var c = this;
      t = Number(t);

      var h,
          d = M.prettyUnit(o),
          f = function (e, n) {
        var r = c.set("date", 1).set(e, n + t);
        return r.set("date", Math.min(c.$D, r.daysInMonth()));
      };

      if (d === a) return f(a, this.$M);
      if (d === u) return f(u, this.$y);

      switch (d) {
        case n:
          h = 6e4;
          break;

        case r:
          h = 36e5;
          break;

        case s:
          h = 864e5;
          break;

        case i:
          h = 6048e5;
          break;

        case e:
          h = 1e3;
          break;

        default:
          h = 1;
      }

      var $ = this.valueOf() + t * h;
      return S($, this);
    }, d.subtract = function (t, e) {
      return this.add(-1 * t, e);
    }, d.format = function (t) {
      var e = this,
          n = t || "YYYY-MM-DDTHH:mm:ssZ",
          r = M.padZoneStr(this.$d.getTimezoneOffset()),
          s = this.$locale(),
          i = s.weekdays,
          a = s.months,
          u = function (t, e, n, r) {
        return t && t[e] || n[e].substr(0, r);
      };

      return n.replace(c, function (t) {
        if (t.indexOf("[") > -1) return t.replace(/\[|\]/g, "");

        switch (t) {
          case "YY":
            return String(e.$y).slice(-2);

          case "YYYY":
            return String(e.$y);

          case "M":
            return String(e.$M + 1);

          case "MM":
            return M.padStart(e.$M + 1, 2, "0");

          case "MMM":
            return u(s.monthsShort, e.$M, a, 3);

          case "MMMM":
            return a[e.$M];

          case "D":
            return String(e.$D);

          case "DD":
            return M.padStart(e.$D, 2, "0");

          case "d":
            return String(e.$W);

          case "dd":
            return u(s.weekdaysMin, e.$W, i, 2);

          case "ddd":
            return u(s.weekdaysShort, e.$W, i, 3);

          case "dddd":
            return i[e.$W];

          case "H":
            return String(e.$H);

          case "HH":
            return M.padStart(e.$H, 2, "0");

          case "h":
          case "hh":
            return 0 === e.$H ? 12 : M.padStart(e.$H < 13 ? e.$H : e.$H - 12, "hh" === t ? 2 : 1, "0");

          case "a":
            return e.$H < 12 ? "am" : "pm";

          case "A":
            return e.$H < 12 ? "AM" : "PM";

          case "m":
            return String(e.$m);

          case "mm":
            return M.padStart(e.$m, 2, "0");

          case "s":
            return String(e.$s);

          case "ss":
            return M.padStart(e.$s, 2, "0");

          case "SSS":
            return M.padStart(e.$ms, 3, "0");

          case "Z":
            return r;

          default:
            return r.replace(":", "");
        }
      });
    }, d.diff = function (t, o, c) {
      var h = M.prettyUnit(o),
          d = p(t),
          f = this - d,
          $ = M.monthDiff(this, d);

      switch (h) {
        case u:
          $ /= 12;
          break;

        case a:
          break;

        case "quarter":
          $ /= 3;
          break;

        case i:
          $ = f / 6048e5;
          break;

        case s:
          $ = f / 864e5;
          break;

        case r:
          $ = f / 36e5;
          break;

        case n:
          $ = f / 6e4;
          break;

        case e:
          $ = f / 1e3;
          break;

        default:
          $ = f;
      }

      return c ? $ : M.absFloor($);
    }, d.daysInMonth = function () {
      return this.endOf(a).$D;
    }, d.$locale = function () {
      return l[this.$L];
    }, d.locale = function (t, e) {
      var n = this.clone();
      return n.$L = y(t, e, !0), n;
    }, d.clone = function () {
      return S(this.toDate(), this);
    }, d.toDate = function () {
      return new Date(this.$d);
    }, d.toArray = function () {
      return [this.$y, this.$M, this.$D, this.$H, this.$m, this.$s, this.$ms];
    }, d.toJSON = function () {
      return this.toISOString();
    }, d.toISOString = function () {
      return this.toDate().toISOString();
    }, d.toObject = function () {
      return {
        years: this.$y,
        months: this.$M,
        date: this.$D,
        hours: this.$H,
        minutes: this.$m,
        seconds: this.$s,
        milliseconds: this.$ms
      };
    }, d.toString = function () {
      return this.$d.toUTCString();
    }, h;
  }();

  return p.extend = function (t, e) {
    return t(e, b, p), p;
  }, p.locale = y, p.isDayjs = m, p.en = l[$], p;
});