"use strict";

Component({
  properties: {
    value: {
      type: null
    },
    checked: {
      type: Boolean,
      value: !1
    },
    disabled: {
      type: Boolean,
      value: !1
    },
    label: {
      type: String
    },
    ripple: {
      type: Boolean,
      value: !0
    },
    reverse: {
      type: Boolean,
      value: !1
    },
    color: {
      type: String,
      value: "#ff4081"
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    checked: !1,
    clicked: !1
  },
  ready: function () {
    this.setData({
      checked: this.properties.checked
    });
  },
  externalClasses: ["sc-class"],
  methods: {
    _changeCheck: function (e) {
      this.setData({
        checked: !this.data.checked,
        clicked: !0
      }), this.triggerEvent("checkchange", {
        checked: this.data.checked,
        value: this.properties.value
      }, {
        bubbles: !0,
        composed: !0
      });
    },
    _animationend: function () {
      this.setData({
        clicked: !1
      });
    }
  }
});