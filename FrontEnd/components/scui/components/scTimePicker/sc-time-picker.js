"use strict";

var dayjs = require("../../assets/lib/day/day"),
    dialogCount = 0;

Component({
  properties: {
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    selectTimeRotate: 0,
    top: 0,
    left: 0,
    baseHour: 3,
    baseMinute: 15,
    date: null,
    dateObject: null,
    AmOrPm: null,
    hourList: [{
      hour: 1,
      active: !1
    }, {
      hour: 2,
      active: !1
    }, {
      hour: 3,
      active: !0
    }, {
      hour: 4,
      active: !1
    }, {
      hour: 5,
      active: !1
    }, {
      hour: 6,
      active: !1
    }, {
      hour: 7,
      active: !1
    }, {
      hour: 8,
      active: !1
    }, {
      hour: 9,
      active: !1
    }, {
      hour: 10,
      active: !1
    }, {
      hour: 11,
      active: !1
    }, {
      hour: 12,
      active: !1
    }],
    minuteList: [{
      minute: 5,
      active: !1
    }, {
      minute: 10,
      active: !1
    }, {
      minute: 15,
      active: !0
    }, {
      minute: 20,
      active: !1
    }, {
      minute: 25,
      active: !1
    }, {
      minute: 30,
      active: !1
    }, {
      minute: 35,
      active: !1
    }, {
      minute: 40,
      active: !1
    }, {
      minute: 45,
      active: !1
    }, {
      minute: 50,
      active: !1
    }, {
      minute: 55,
      active: !1
    }, {
      minute: 0,
      active: !1
    }],
    minuteView: !1,
    defaultOption: {}
  },
  externalClasses: ["sc-class"],
  ready: function () {
    this.setData({
      dialogCount: dialogCount++
    }), this.data.dialog = this.selectComponent("#time-picker-dialog-" + this.data.dialogCount);
  },
  methods: {
    hourtouchmove: function (t) {
      var e = this;
      this.getAngle(t, ".time-hour").then(function (t) {
        var i = Math.ceil(t / 15);
        i -= i % 2 == 0 ? 0 : 1;
        var a = i / 2,
            n = e.data.date.hour();

        if (0 === n ? a = 12 === a || 0 === a ? 0 : a : 12 === n && (a = 0 === a || 12 === a ? 12 : a), a !== n && a + 12 !== n) {
          0 === n ? 11 === a && (a += 12) : 12 === n ? 1 === a && (a += 12) : "PM" === e.data.AmOrPm && (a += 12);
          var o = e.data.date.set("hour", a),
              s = o.toObject();
          s.minutes;
          e.setData({
            date: o,
            dateObject: s,
            selectTimeRotate: e.getAngleByHour(a),
            AmOrPm: e.getAmOrPm(o)
          });
        }
      });
    },
    minutetouchmove: function (t) {
      var e = this;
      this.getAngle(t, ".time-minute").then(function (t) {
        var i = t,
            a = Math.ceil(i / 3);

        if ((i = 3 * (a - (a % 2 == 0 ? 0 : 1))) !== e.data.selectTimeRotate) {
          var n = e.getMinute(i);
          n = (n > 0 ? n : 60 + n) % 60;
          var o = dayjs(e.data.date.set("minute", n).toISOString()),
              s = o.toObject(),
              u = s.minutes;
          s.minutesShow = u.toString().length >= 2 ? u : "0" + u, e.setData({
            date: o,
            dateObject: s,
            selectTimeRotate: i
          });
        }
      });
    },
    hourtap: function (t) {
      var e = this;
      this.getAngle(t, ".time-hour").then(function (t) {
        var i = Math.ceil(t / 15);
        i -= i % 2 == 0 ? 0 : 1;
        var a = i / 2;
        "PM" === e.data.AmOrPm && (a += 12);
        var n = e.data.date.set("hour", a),
            o = n.toObject(),
            s = o.minutes;
        o.minutesShow = s.toString().length >= 2 ? s : "0" + s, e.setData({
          date: n,
          dateObject: o,
          selectTimeRotate: e.getAngleByHour(a)
        });
      });
    },
    minutetap: function (t) {
      this.minutetouchmove(t);
    },
    changeView: function () {
      var t = this.data.minuteView,
          e = this.data.date;
      this.setData({
        minuteView: !t,
        selectTimeRotate: t ? this.getAngleByHour(e.hour()) : this.getAngleByMinute(e.minute())
      });
    },
    changeViewToHour: function () {
      var t = this.data.date;
      this.setData({
        minuteView: !1,
        selectTimeRotate: this.getAngleByHour(t.hour())
      });
    },
    changeViewToMinute: function () {
      var t = this.data.date;
      this.setData({
        minuteView: !0,
        selectTimeRotate: this.getAngleByMinute(t.minute())
      });
    },
    changeClockToAM: function () {
      var t = this.data.date,
          e = t.hour();
      e >= 12 && (e -= 12), t = t.set("hour", e);
      var i = t.toObject(),
          a = i.minutes;
      i.minutesShow = a.toString().length >= 2 ? a : "0" + a, this.setData({
        date: t,
        dateObject: i,
        AmOrPm: this.getAmOrPm(t)
      }), this.data.minuteView || this.setData({
        selectTimeRotate: this.getAngleByHour(e)
      });
    },
    changeClockToPM: function () {
      var t = this.data.date,
          e = t.hour();
      e <= 12 && (e += 12), t = t.set("hour", e);
      var i = t.toObject(),
          a = i.minutes;
      i.minutesShow = a.toString().length >= 2 ? a : "0" + a, this.setData({
        date: t,
        dateObject: i,
        AmOrPm: this.getAmOrPm(t)
      }), this.data.minuteView || this.setData({
        selectTimeRotate: this.getAngleByHour(e)
      });
    },
    getAmOrPm: function (t) {
      return t.hour() > 12 ? "PM" : "AM";
    },
    getHour: function (t) {
      return "PM" === this.data.AmOrPm ? t / 30 + 12 : t / 30;
    },
    getMinute: function (t) {
      return t / 6;
    },
    getAngleByHour: function (t) {
      return 30 * t;
    },
    getAngleByMinute: function (t) {
      return 6 * t;
    },
    _queryMultipleNodes: function (t) {
      var e = this;
      return new Promise(function (i) {
        var a = e.createSelectorQuery();
        a.select(t).boundingClientRect(), a.exec(function (t) {
          i(t);
        });
      });
    },
    getAngle: function (t, e) {
      var i = this;
      return new Promise(function (a) {
        i._queryMultipleNodes(e).then(function (e) {
          var i = e[0],
              n = i.top,
              o = i.left,
              s = i.width,
              u = t.changedTouches[0],
              r = u.clientX,
              c = u.clientY,
              h = o + s / 2,
              m = n + s / 2,
              l = 90 - 180 * Math.atan2(m - c, r - h) / Math.PI;
          a(parseInt((l > 0 ? l : 360 + l).toString()));
        });
      });
    },
    _open: function (t) {
      var e = JSON.parse(JSON.stringify(this.data.defaultOption)),
          i = Object.assign(e, t);
      i.date = i.date || new Date();
      var a = dayjs(i.date),
          n = a.toObject(),
          o = n.minutes;
      n.minutesShow = o.toString().length >= 2 ? o : "0" + o, this.setData({
        date: a,
        dateObject: n,
        selectTimeRotate: this.getAngleByHour(a.hour()),
        AmOrPm: this.getAmOrPm(a)
      }), this.data.dialog._open();
    },
    _close: function () {
      this.data.dialog._close();
    },
    _submit: function () {
      var t = this.data.dateObject,
          e = t.years,
          i = t.months,
          a = t.date,
          n = t.hours,
          o = t.minutes,
          s = t.seconds,
          u = t.milliseconds,
          r = dayjs();
      r = r.set("year", e).set("month", i).set("date", a).set("hour", n).set("minute", o).set("second", s).set("millisecond", u), this.triggerEvent("submit", {
        value: r.toDate()
      }), this._close();
    },
    dialogOpen: function () {
      this.triggerEvent("open", {
        bubbles: !0
      });
    },
    dialogClose: function () {
      this.triggerEvent("close", {
        bubbles: !0
      });
    },
    dialogOpened: function () {
      this.triggerEvent("opened", {
        bubbles: !0
      });
    },
    dialogClosed: function () {
      this.triggerEvent("closed", {
        bubbles: !0
      });
    }
  }
});