"use strict";

Component({
  properties: {
    type: {
      type: String,
      value: "indeterminate"
    },
    width: {
      type: Number,
      value: 0
    },
    color: {
      type: String,
      value: "rgb(63, 81, 181);"
    },
    size: {
      type: String,
      value: "8"
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  externalClasses: ["sc-class", "sc-determinate-class"],
  data: {},
  methods: {}
});