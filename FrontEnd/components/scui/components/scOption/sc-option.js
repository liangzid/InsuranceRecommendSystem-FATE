"use strict";

Component({
  properties: {
    value: {
      type: null
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    checked: !1,
    clicked: !1,
    showRipple: !1,
    disabled: !1,
    value: null
  },
  ready: function () {
    this.setData({
      checked: this.properties.checked,
      disabled: this.properties.disabled,
      value: this.properties.value
    });
  },
  externalClasses: ["sc-class"],
  methods: {}
});