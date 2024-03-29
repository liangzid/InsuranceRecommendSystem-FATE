"use strict";

Component({
  properties: {
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    show: !1,
    timer: null,
    defaultOption: {
      timeout: 4e3,
      position: "bottom",
      buttonText: "",
      buttonTextColor: "#90CAF9",
      closeOnButtonClick: !1,
      onClick: null,
      onButtonClick: null,
      onOpen: null,
      onOpened: null,
      onClose: null,
      onClosed: null
    },
    snackBarStack: []
  },
  ready: function () {},
  externalClasses: ["sc-class", "sc-button-class"],
  methods: {
    _open: function (t) {
      var n = JSON.parse(JSON.stringify(this.data.defaultOption)),
          o = Object.assign(n, t);
      this.data.snackBarStack.push(o), this._openNext();
    },
    _close: function () {
      var t = this;
      this.setData({
        show: !1
      });
      var n = this.data.options.onClose;
      n && "function" == typeof n && n(), setTimeout(function () {
        var n = t.data.options.onClosed;
        n && "function" == typeof n && n(), t.data.snackBarStack.shift(), t.data.snackBarStack.length > 0 && t._openNext();
      }, 300);
    },
    _openNext: function () {
      var t = this,
          n = this.data.snackBarStack;

      if (n.length > 0 && !this.data.show) {
        var o = n[0];
        this.setData({
          options: o,
          show: !0,
          startTime: Date.parse(new Date())
        });
        var a = this.data.options,
            e = a.onOpen,
            s = a.onOpened;
        e && "function" == typeof e && e(), setTimeout(function () {
          s && "function" == typeof s && s();
        }, 300), this.data.timer = setTimeout(function () {
          t._close();
        }, o.timeout);
      }
    },
    _btnTap: function () {
      var t = this,
          n = this.data.options,
          o = n.closeOnButtonClick,
          a = n.onButtonClick;
      n.timeout;
      if (clearTimeout(this.data.timer), o) this._close();else {
        var e = e - (Date.parse(new Date()) - this.data.startTime);
        this.data.timer = setTimeout(function () {
          t._close();
        }, (e > 0 ? e : 0) + 1e3);
      }
      a && "function" == typeof a && a();
    },
    _snackBar: function () {
      var t = this.data.options.onClick;
      t && "function" == typeof t && t();
    }
  }
});