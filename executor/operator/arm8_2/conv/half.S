/* float16_t type_info */ 
.weak _ZTIDh
.section .data.rel.ro._ZTIDh,"awG",@progbits,_ZTIDh,comdat
.align 8
.type _ZTIDh, @object
.size _ZTIDh, 16
_ZTIDh:
.quad _ZTVN10__cxxabiv117__class_type_infoE+16
.quad _ZTSDh

.weak _ZTSDh
.section .rodata._ZTSDh,"aG",@progbits,_ZTSDh,comdat
.align 8
.type _ZTSDh, @object

_ZTSDh:
.string "half"

/* float16_t* type_info */ 
.weak _ZTIPDh
.section .data.rel.ro._ZTIPDh,"awG",@progbits,_ZTIPDh,comdat
.align 8
.type _ZTIPDh, @object
.size _ZTIPDh, 16
_ZTIPDh:
.quad _ZTVN10__cxxabiv117__class_type_infoE+16
.quad _ZTSPDh

.weak _ZTSPDh
.section .rodata._ZTSPDh,"aG",@progbits,_ZTSPDh,comdat
.align 8
.type _ZTSPDh, @object

_ZTSPDh:
.string "half"