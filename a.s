	.file	"main.cpp"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"basic_string::_M_construct null not valid"
	.text
	.align 2
	.p2align 4,,15
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag.constprop.174, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag.constprop.174:
.LFB4251:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	subq	$16, %rsp
	.cfi_def_cfa_offset 48
	movq	%fs:40, %rax
	movq	%rax, 8(%rsp)
	xorl	%eax, %eax
	testq	%rdx, %rdx
	je	.L2
	testq	%rsi, %rsi
	je	.L20
.L2:
	movq	%rdx, %rbx
	subq	%r12, %rbx
	movq	%rbx, (%rsp)
	cmpq	$15, %rbx
	ja	.L21
	movq	0(%rbp), %rdx
	movq	%rdx, %rax
	cmpq	$1, %rbx
	jne	.L5
	movzbl	(%r12), %eax
	movb	%al, (%rdx)
	movq	0(%rbp), %rdx
.L6:
	movq	(%rsp), %rax
	movq	%rax, 8(%rbp)
	movb	$0, (%rdx,%rax)
	movq	8(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L22
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L5:
	.cfi_restore_state
	testq	%rbx, %rbx
	je	.L6
	jmp	.L4
.L21:
	xorl	%edx, %edx
	movq	%rsp, %rsi
	movq	%rbp, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@PLT
	movq	(%rsp), %rdx
	movq	%rax, 0(%rbp)
	movq	%rdx, 16(%rbp)
.L4:
	movq	%rbx, %rdx
	movq	%r12, %rsi
	movq	%rax, %rdi
	call	memcpy@PLT
	movq	0(%rbp), %rdx
	jmp	.L6
.L20:
	leaq	.LC0(%rip), %rdi
	call	_ZSt19__throw_logic_errorPKc@PLT
.L22:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE4251:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag.constprop.174, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag.constprop.174
	.p2align 4,,15
	.type	_ZSt13__adjust_heapIPSt4pairIdiElS1_N9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S7_T1_T2_.constprop.178, @function
_ZSt13__adjust_heapIPSt4pairIdiElS1_N9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S7_T1_T2_.constprop.178:
.LFB4248:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	leaq	-1(%rdx), %rax
	movq	%rax, %r11
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	shrq	$63, %r11
	addq	%rax, %r11
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	sarq	%r11
	movq	%rdx, %rbx
	andl	$1, %ebx
	movl	%ecx, %ebp
	cmpq	%r11, %rsi
	jge	.L24
	movq	%rsi, %r8
	.p2align 4,,10
	.p2align 3
.L28:
	leaq	1(%r8), %rax
	leaq	(%rax,%rax), %r9
	salq	$5, %rax
	leaq	-16(%rdi,%rax), %r12
	addq	%rdi, %rax
	vmovsd	(%rax), %xmm1
	vmovsd	(%r12), %xmm2
	vcomisd	%xmm1, %xmm2
	ja	.L25
	movl	8(%rax), %r10d
	jne	.L26
	cmpl	%r10d, 8(%r12)
	jle	.L26
.L25:
	decq	%r9
	movq	%r9, %rax
	salq	$4, %rax
	addq	%rdi, %rax
	vmovsd	(%rax), %xmm1
	movl	8(%rax), %r10d
.L26:
	salq	$4, %r8
	addq	%rdi, %r8
	vmovsd	%xmm1, (%r8)
	movl	%r10d, 8(%r8)
	movq	%r9, %r8
	cmpq	%r9, %r11
	jg	.L28
	testq	%rbx, %rbx
	je	.L35
.L29:
	leaq	-1(%r9), %r8
	movq	%r8, %rdx
	shrq	$63, %rdx
	addq	%r8, %rdx
	sarq	%rdx
	cmpq	%rsi, %r9
	jle	.L30
.L34:
	movq	%rdx, %rax
	salq	$4, %rax
	addq	%rdi, %rax
	vmovsd	(%rax), %xmm1
	vcomisd	%xmm1, %xmm0
	ja	.L31
	vcomisd	%xmm0, %xmm1
	jne	.L44
	movl	8(%rax), %r8d
	cmpl	%ebp, %r8d
	jge	.L44
.L33:
	salq	$4, %r9
	addq	%rdi, %r9
	movl	%r8d, 8(%r9)
	vmovsd	%xmm1, (%r9)
	leaq	-1(%rdx), %r9
	movq	%r9, %r8
	shrq	$63, %r8
	addq	%r9, %r8
	sarq	%r8
	movq	%rdx, %r9
	cmpq	%rdx, %rsi
	jge	.L30
	movq	%r8, %rdx
	jmp	.L34
	.p2align 4,,10
	.p2align 3
.L44:
	movq	%r9, %rax
	salq	$4, %rax
	addq	%rdi, %rax
.L30:
	movl	%ecx, 8(%rax)
	vmovsd	%xmm0, (%rax)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L31:
	.cfi_restore_state
	movl	8(%rax), %r8d
	jmp	.L33
	.p2align 4,,10
	.p2align 3
.L24:
	movq	%rsi, %rax
	salq	$4, %rax
	addq	%rdi, %rax
	movq	%rsi, %r9
	testq	%rbx, %rbx
	jne	.L30
	.p2align 4,,10
	.p2align 3
.L35:
	leaq	-2(%rdx), %r8
	movq	%r8, %rdx
	shrq	$63, %rdx
	addq	%r8, %rdx
	sarq	%rdx
	cmpq	%r9, %rdx
	jne	.L29
	incq	%r9
	movq	%r9, %rdx
	salq	$5, %rdx
	leaq	-16(%rdi,%rdx), %rdx
	vmovsd	(%rdx), %xmm1
	movl	8(%rdx), %edx
	leaq	-1(%r9,%r9), %r9
	vmovsd	%xmm1, (%rax)
	movl	%edx, 8(%rax)
	movq	%r9, %rax
	salq	$4, %rax
	addq	%rdi, %rax
	jmp	.L29
	.cfi_endproc
.LFE4248:
	.size	_ZSt13__adjust_heapIPSt4pairIdiElS1_N9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S7_T1_T2_.constprop.178, .-_ZSt13__adjust_heapIPSt4pairIdiElS1_N9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S7_T1_T2_.constprop.178
	.p2align 4,,15
	.type	_ZSt16__insertion_sortIPSt4pairIdiEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_.constprop.180, @function
_ZSt16__insertion_sortIPSt4pairIdiEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_.constprop.180:
.LFB4246:
	.cfi_startproc
	cmpq	%rsi, %rdi
	je	.L70
	leaq	16(%rdi), %rax
	cmpq	%rax, %rsi
	jne	.L59
	jmp	.L70
	.p2align 4,,10
	.p2align 3
.L71:
	vcomisd	%xmm0, %xmm1
	jne	.L61
	cmpl	%r9d, 8(%rdi)
	jg	.L48
.L61:
	movq	%rax, %rdx
.L58:
	vmovsd	-16(%rdx), %xmm0
	vcomisd	%xmm1, %xmm0
	ja	.L54
	vcomisd	%xmm0, %xmm1
	jne	.L55
	movl	-8(%rdx), %ecx
	cmpl	%r9d, %ecx
	jg	.L57
.L55:
	leaq	16(%rax), %r8
	vmovsd	%xmm1, (%rdx)
	movl	%r9d, 8(%rdx)
	movq	%r8, %rax
	cmpq	%r8, %rsi
	je	.L70
.L59:
	vmovsd	(%rax), %xmm1
	vmovsd	(%rdi), %xmm0
	movl	8(%rax), %r9d
	vcomisd	%xmm1, %xmm0
	jbe	.L71
.L48:
	leaq	16(%rax), %r8
	subq	%rdi, %rax
	movq	%rax, %rdx
	movq	%rax, %rcx
	sarq	$4, %rdx
	movq	%r8, %rax
	testq	%rcx, %rcx
	jle	.L53
	.p2align 4,,10
	.p2align 3
.L51:
	subq	$16, %rax
	vmovsd	-16(%rax), %xmm0
	movl	-8(%rax), %ecx
	vmovsd	%xmm0, (%rax)
	movl	%ecx, 8(%rax)
	decq	%rdx
	jne	.L51
.L53:
	vmovsd	%xmm1, (%rdi)
	movl	%r9d, 8(%rdi)
	movq	%r8, %rax
	cmpq	%r8, %rsi
	jne	.L59
.L70:
	ret
	.p2align 4,,10
	.p2align 3
.L54:
	movl	-8(%rdx), %ecx
.L57:
	vmovsd	%xmm0, (%rdx)
	movl	%ecx, 8(%rdx)
	subq	$16, %rdx
	jmp	.L58
	.cfi_endproc
.LFE4246:
	.size	_ZSt16__insertion_sortIPSt4pairIdiEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_.constprop.180, .-_ZSt16__insertion_sortIPSt4pairIdiEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_.constprop.180
	.p2align 4,,15
	.type	_ZSt16__introsort_loopIPSt4pairIdiElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_T1_.isra.144, @function
_ZSt16__introsort_loopIPSt4pairIdiElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_T1_.isra.144:
.LFB4218:
	.cfi_startproc
	movq	%rsi, %rax
	subq	%rdi, %rax
	cmpq	$256, %rax
	jle	.L132
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	movq	%rdx, %r14
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	leaq	16(%rdi), %r12
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	%rdi, %rbx
	testq	%rdx, %rdx
	je	.L135
.L76:
	sarq	$5, %rax
	salq	$4, %rax
	addq	%rbx, %rax
	vmovsd	16(%rbx), %xmm0
	vmovsd	(%rax), %xmm1
	decq	%r14
	vcomisd	%xmm0, %xmm1
	ja	.L80
	vcomisd	%xmm1, %xmm0
	jne	.L81
	movl	8(%rax), %edi
	cmpl	%edi, 24(%rbx)
	jl	.L80
.L81:
	vmovsd	-16(%rsi), %xmm2
	vcomisd	%xmm0, %xmm2
	jbe	.L136
	movl	24(%rbx), %edx
.L98:
	movl	8(%rbx), %eax
	vmovsd	(%rbx), %xmm1
	movl	%edx, 8(%rbx)
	movl	%eax, 24(%rbx)
	vmovsd	%xmm0, (%rbx)
	vmovsd	%xmm1, 16(%rbx)
.L95:
	movq	%r12, %rbp
	movq	%rsi, %rax
	jmp	.L91
	.p2align 4,,10
	.p2align 3
.L128:
	movl	8(%rbp), %edx
	movl	8(%rax), %ecx
	vmovsd	%xmm2, 0(%rbp)
	vmovsd	%xmm1, (%rax)
	movl	%ecx, 8(%rbp)
	movl	%edx, 8(%rax)
	vmovsd	(%rbx), %xmm0
.L104:
	addq	$16, %rbp
.L91:
	vmovsd	0(%rbp), %xmm1
	movq	%rbp, %r13
	vcomisd	%xmm1, %xmm0
	ja	.L104
	vcomisd	%xmm0, %xmm1
	jne	.L105
	movl	8(%rbx), %edi
	cmpl	%edi, 8(%rbp)
	jl	.L104
	.p2align 4,,10
	.p2align 3
.L105:
	subq	$16, %rax
	vmovsd	(%rax), %xmm2
	vcomisd	%xmm0, %xmm2
	ja	.L105
	jne	.L108
	movl	8(%rax), %edi
	cmpl	%edi, 8(%rbx)
	jl	.L105
	vmovapd	%xmm0, %xmm2
.L108:
	cmpq	%rax, %rbp
	jb	.L128
	movq	%r14, %rdx
	movq	%rbp, %rdi
	call	_ZSt16__introsort_loopIPSt4pairIdiElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_T1_.isra.144
	movq	%rbp, %rax
	subq	%rbx, %rax
	cmpq	$256, %rax
	jle	.L130
	movq	%rbp, %rsi
	testq	%r14, %r14
	jne	.L76
.L74:
	sarq	$4, %rax
	leaq	-2(%rax), %r12
	sarq	%r12
	movq	%r12, %rdx
	salq	$4, %rdx
	movq	%rax, %rbp
	movq	(%rbx,%rdx), %rax
	movq	8(%rbx,%rdx), %rcx
	vmovq	%rax, %xmm0
	movq	%rbp, %rdx
	movq	%r12, %rsi
	movq	%rbx, %rdi
	call	_ZSt13__adjust_heapIPSt4pairIdiElS1_N9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S7_T1_T2_.constprop.178
.L77:
	decq	%r12
	movq	%r12, %rdx
	salq	$4, %rdx
	movq	(%rbx,%rdx), %rax
	movq	8(%rbx,%rdx), %rcx
	vmovq	%rax, %xmm0
	movq	%rbp, %rdx
	movq	%r12, %rsi
	movq	%rbx, %rdi
	call	_ZSt13__adjust_heapIPSt4pairIdiElS1_N9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S7_T1_T2_.constprop.178
	testq	%r12, %r12
	jne	.L77
	.p2align 4,,10
	.p2align 3
.L78:
	subq	$16, %r13
	movl	8(%rbx), %edx
	movq	%r13, %rbp
	vmovsd	(%rbx), %xmm0
	movq	0(%r13), %rax
	subq	%rbx, %rbp
	movq	8(%r13), %rcx
	movl	%edx, 8(%r13)
	movq	%rbp, %rdx
	vmovsd	%xmm0, 0(%r13)
	sarq	$4, %rdx
	xorl	%esi, %esi
	vmovq	%rax, %xmm0
	movq	%rbx, %rdi
	call	_ZSt13__adjust_heapIPSt4pairIdiElS1_N9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S7_T1_T2_.constprop.178
	cmpq	$16, %rbp
	jg	.L78
.L130:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L80:
	.cfi_restore_state
	vmovsd	-16(%rsi), %xmm2
	vcomisd	%xmm1, %xmm2
	ja	.L85
	vcomisd	%xmm2, %xmm1
	jne	.L86
	movl	8(%rax), %edx
	cmpl	-8(%rsi), %edx
	jge	.L86
.L88:
	vmovsd	(%rbx), %xmm0
	movl	8(%rbx), %ecx
	vmovsd	%xmm1, (%rbx)
	vmovsd	%xmm0, (%rax)
	vmovsd	(%rbx), %xmm0
	movl	%edx, 8(%rbx)
	movl	%ecx, 8(%rax)
	jmp	.L95
	.p2align 4,,10
	.p2align 3
.L86:
	vcomisd	%xmm0, %xmm2
	jbe	.L137
	movl	-8(%rsi), %eax
.L94:
	vmovsd	(%rbx), %xmm0
	movl	8(%rbx), %edx
	vmovsd	%xmm2, (%rbx)
	vmovsd	%xmm0, -16(%rsi)
	vmovsd	(%rbx), %xmm0
	movl	%eax, 8(%rbx)
	movl	%edx, -8(%rsi)
	jmp	.L95
	.p2align 4,,10
	.p2align 3
.L85:
	movl	8(%rax), %edx
	jmp	.L88
	.p2align 4,,10
	.p2align 3
.L136:
	vcomisd	%xmm2, %xmm0
	jne	.L96
	movl	24(%rbx), %edx
	cmpl	-8(%rsi), %edx
	jl	.L98
.L96:
	vcomisd	%xmm1, %xmm2
	jbe	.L138
	movl	-8(%rsi), %edx
.L103:
	vmovsd	(%rbx), %xmm0
	movl	8(%rbx), %eax
	vmovsd	%xmm2, (%rbx)
	vmovsd	%xmm0, -16(%rsi)
	vmovsd	(%rbx), %xmm0
	movl	%edx, 8(%rbx)
	movl	%eax, -8(%rsi)
	jmp	.L95
	.p2align 4,,10
	.p2align 3
.L137:
	vcomisd	%xmm2, %xmm0
	movl	24(%rbx), %edx
	jne	.L98
	movl	-8(%rsi), %eax
	cmpl	%edx, %eax
	jg	.L94
	jmp	.L98
	.p2align 4,,10
	.p2align 3
.L138:
	vcomisd	%xmm2, %xmm1
	movl	8(%rax), %ecx
	jne	.L101
	movl	-8(%rsi), %edx
	cmpl	%ecx, %edx
	jg	.L103
.L101:
	vmovsd	(%rbx), %xmm0
	movl	8(%rbx), %edx
	vmovsd	%xmm1, (%rbx)
	vmovsd	%xmm0, (%rax)
	vmovsd	(%rbx), %xmm0
	movl	%ecx, 8(%rbx)
	movl	%edx, 8(%rax)
	jmp	.L95
	.p2align 4,,10
	.p2align 3
.L132:
	.cfi_def_cfa_offset 8
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	ret
	.p2align 4,,10
	.p2align 3
.L135:
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	.cfi_offset 6, -40
	.cfi_offset 12, -32
	.cfi_offset 13, -24
	.cfi_offset 14, -16
	movq	%rsi, %r13
	jmp	.L74
	.cfi_endproc
.LFE4218:
	.size	_ZSt16__introsort_loopIPSt4pairIdiElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_T1_.isra.144, .-_ZSt16__introsort_loopIPSt4pairIdiElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_T1_.isra.144
	.section	.text._ZN8KnnModelD2Ev,"axG",@progbits,_ZN8KnnModelD5Ev,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZN8KnnModelD2Ev
	.type	_ZN8KnnModelD2Ev, @function
_ZN8KnnModelD2Ev:
.LFB2872:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	72(%rdi), %rdi
	testq	%rdi, %rdi
	je	.L140
	call	_ZdlPv@PLT
.L140:
	movq	56(%rbp), %r12
	movq	48(%rbp), %rbx
	cmpq	%rbx, %r12
	je	.L141
	.p2align 4,,10
	.p2align 3
.L145:
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.L142
	addq	$24, %rbx
	call	_ZdlPv@PLT
	cmpq	%rbx, %r12
	jne	.L145
.L143:
	movq	48(%rbp), %rbx
.L141:
	testq	%rbx, %rbx
	je	.L146
	movq	%rbx, %rdi
	call	_ZdlPv@PLT
.L146:
	movq	24(%rbp), %r12
	movq	16(%rbp), %rbx
	cmpq	%rbx, %r12
	je	.L147
	.p2align 4,,10
	.p2align 3
.L151:
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.L148
	addq	$24, %rbx
	call	_ZdlPv@PLT
	cmpq	%rbx, %r12
	jne	.L151
.L149:
	movq	16(%rbp), %rbx
.L147:
	testq	%rbx, %rbx
	je	.L159
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	jmp	_ZdlPv@PLT
	.p2align 4,,10
	.p2align 3
.L148:
	.cfi_restore_state
	addq	$24, %rbx
	cmpq	%rbx, %r12
	jne	.L151
	jmp	.L149
	.p2align 4,,10
	.p2align 3
.L142:
	addq	$24, %rbx
	cmpq	%rbx, %r12
	jne	.L145
	jmp	.L143
	.p2align 4,,10
	.p2align 3
.L159:
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE2872:
	.size	_ZN8KnnModelD2Ev, .-_ZN8KnnModelD2Ev
	.weak	_ZN8KnnModelD1Ev
	.set	_ZN8KnnModelD1Ev,_ZN8KnnModelD2Ev
	.text
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModelC2Ev
	.type	_ZN8KnnModelC2Ev, @function
_ZN8KnnModelC2Ev:
.LFB2883:
	.cfi_startproc
	movq	$0, 16(%rdi)
	movq	$0, 24(%rdi)
	movq	$0, 32(%rdi)
	movq	$0, 48(%rdi)
	movq	$0, 56(%rdi)
	movq	$0, 64(%rdi)
	movq	$0, 72(%rdi)
	movq	$0, 80(%rdi)
	movq	$0, 88(%rdi)
	ret
	.cfi_endproc
.LFE2883:
	.size	_ZN8KnnModelC2Ev, .-_ZN8KnnModelC2Ev
	.globl	_ZN8KnnModelC1Ev
	.set	_ZN8KnnModelC1Ev,_ZN8KnnModelC2Ev
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel12SetAlgorithmE9Algorithm
	.type	_ZN8KnnModel12SetAlgorithmE9Algorithm, @function
_ZN8KnnModel12SetAlgorithmE9Algorithm:
.LFB2886:
	.cfi_startproc
	movl	%esi, 40(%rdi)
	ret
	.cfi_endproc
.LFE2886:
	.size	_ZN8KnnModel12SetAlgorithmE9Algorithm, .-_ZN8KnnModel12SetAlgorithmE9Algorithm
	.section	.rodata.str1.8
	.align 8
.LC1:
	.string	"This instance has not been solved!"
	.section	.text.unlikely,"ax",@progbits
	.align 2
.LCOLDB2:
	.text
.LHOTB2:
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.type	_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, @function
_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE:
.LFB2888:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2888
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rdi, %r15
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, %rbx
	subq	$568, %rsp
	.cfi_def_cfa_offset 624
	leaq	32(%rsp), %rbp
	leaq	248(%rbp), %rdi
	movq	%fs:40, %rax
	movq	%rax, 552(%rsp)
	xorl	%eax, %eax
	call	_ZNSt8ios_baseC2Ev@PLT
	xorl	%ecx, %ecx
	leaq	16+_ZTVSt9basic_iosIcSt11char_traitsIcEE(%rip), %rax
	movw	%cx, 504(%rsp)
	movq	%rax, 280(%rsp)
	movq	8+_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	$0, 496(%rsp)
	movq	-24(%rax), %rdi
	movq	%rax, 32(%rsp)
	movq	16+_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE(%rip), %rax
	addq	%rbp, %rdi
	movq	$0, 512(%rsp)
	movq	$0, 520(%rsp)
	movq	$0, 528(%rsp)
	movq	$0, 536(%rsp)
	movq	%rax, (%rdi)
	xorl	%esi, %esi
.LEHB0:
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@PLT
.LEHE0:
	leaq	24+_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 32(%rsp)
	leaq	8(%rbp), %rdi
	addq	$40, %rax
	movq	%rax, 280(%rsp)
.LEHB1:
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@PLT
.LEHE1:
	leaq	8(%rbp), %rsi
	leaq	248(%rbp), %rdi
.LEHB2:
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@PLT
	movq	(%rbx), %rsi
	leaq	8(%rbp), %rdi
	movl	$16, %edx
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@PLT
	movq	32(%rsp), %rdx
	movq	-24(%rdx), %rdi
	addq	%rbp, %rdi
	testq	%rax, %rax
	je	.L191
	xorl	%esi, %esi
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
.LEHE2:
.L165:
	movq	48(%r15), %rax
	cmpq	56(%r15), %rax
	je	.L192
	movl	(%r15), %edx
	testl	%edx, %edx
	jle	.L171
	leaq	31(%rsp), %rcx
	movq	%rcx, 8(%rsp)
	xorl	%r14d, %r14d
	leaq	30(%rsp), %r13
	.p2align 4,,10
	.p2align 3
.L176:
	leaq	(%r14,%r14,2), %rdx
	leaq	(%rax,%rdx,8), %rax
	movq	8(%rax), %r12
	movq	(%rax), %rbx
	cmpq	%rbx, %r12
	je	.L175
	.p2align 4,,10
	.p2align 3
.L174:
	movl	(%rbx), %esi
	movq	%rbp, %rdi
.LEHB3:
	call	_ZNSolsEi@PLT
	movl	$1, %edx
	movq	%r13, %rsi
	movq	%rax, %rdi
	movb	$32, 30(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	addq	$4, %rbx
	cmpq	%rbx, %r12
	jne	.L174
.L175:
	movq	8(%rsp), %rsi
	movl	$1, %edx
	movq	%rbp, %rdi
	movb	$10, 31(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	incq	%r14
	cmpl	%r14d, (%r15)
	jle	.L171
	movq	48(%r15), %rax
	jmp	.L176
.L192:
	movl	$34, %edx
	leaq	.LC1(%rip), %rsi
	movq	%rbp, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	.p2align 4,,10
	.p2align 3
.L171:
	leaq	8(%rbp), %rdi
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@PLT
.LEHE3:
	testq	%rax, %rax
	je	.L193
.L177:
	leaq	24+_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 32(%rsp)
	addq	$40, %rax
	movq	%rax, 280(%rsp)
	leaq	8(%rbp), %rdi
	leaq	16+_ZTVSt13basic_filebufIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 40(%rsp)
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@PLT
	leaq	112(%rbp), %rdi
	call	_ZNSt12__basic_fileIcED1Ev@PLT
	leaq	16+_ZTVSt15basic_streambufIcSt11char_traitsIcEE(%rip), %rax
	leaq	64(%rbp), %rdi
	movq	%rax, 40(%rsp)
	call	_ZNSt6localeD1Ev@PLT
	movq	8+_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	16+_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE(%rip), %rcx
	movq	%rax, 32(%rsp)
	movq	-24(%rax), %rax
	leaq	248(%rbp), %rdi
	movq	%rcx, 32(%rsp,%rax)
	leaq	16+_ZTVSt9basic_iosIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 280(%rsp)
	call	_ZNSt8ios_baseD2Ev@PLT
	movq	552(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L194
	addq	$568, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L191:
	.cfi_restore_state
	movl	32(%rdi), %esi
	orl	$4, %esi
.LEHB4:
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
.LEHE4:
	jmp	.L165
.L193:
	movq	32(%rsp), %rax
	movq	-24(%rax), %rdi
	addq	%rbp, %rdi
	movl	32(%rdi), %esi
	orl	$4, %esi
.LEHB5:
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
.LEHE5:
	jmp	.L177
.L194:
	call	__stack_chk_fail@PLT
.L180:
	movq	%rax, %rbx
	jmp	.L178
.L183:
	movq	%rax, %rbx
	jmp	.L168
.L182:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L169
.L181:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L170
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA2888:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2888-.LLSDACSB2888
.LLSDACSB2888:
	.uleb128 .LEHB0-.LFB2888
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L181-.LFB2888
	.uleb128 0
	.uleb128 .LEHB1-.LFB2888
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L182-.LFB2888
	.uleb128 0
	.uleb128 .LEHB2-.LFB2888
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L183-.LFB2888
	.uleb128 0
	.uleb128 .LEHB3-.LFB2888
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L180-.LFB2888
	.uleb128 0
	.uleb128 .LEHB4-.LFB2888
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L183-.LFB2888
	.uleb128 0
	.uleb128 .LEHB5-.LFB2888
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L180-.LFB2888
	.uleb128 0
.LLSDACSE2888:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC2888
	.type	_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE.cold.181, @function
_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE.cold.181:
.LFSB2888:
.L178:
	.cfi_def_cfa_offset 624
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	.cfi_offset 12, -40
	.cfi_offset 13, -32
	.cfi_offset 14, -24
	.cfi_offset 15, -16
	movq	%rbp, %rdi
	vzeroupper
	call	_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@PLT
	movq	%rbx, %rdi
.LEHB6:
	call	_Unwind_Resume@PLT
.L168:
	leaq	8(%rbp), %rdi
	vzeroupper
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@PLT
.L169:
	movq	8+_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 32(%rsp)
	movq	-24(%rax), %rdx
	movq	16+_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 32(%rsp,%rdx)
.L170:
	leaq	16+_ZTVSt9basic_iosIcSt11char_traitsIcEE(%rip), %rax
	leaq	248(%rbp), %rdi
	movq	%rax, 280(%rsp)
	call	_ZNSt8ios_baseD2Ev@PLT
	movq	%rbx, %rdi
	call	_Unwind_Resume@PLT
.LEHE6:
	.cfi_endproc
.LFE2888:
	.section	.gcc_except_table
.LLSDAC2888:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC2888-.LLSDACSBC2888
.LLSDACSBC2888:
	.uleb128 .LEHB6-.LCOLDB2
	.uleb128 .LEHE6-.LEHB6
	.uleb128 0
	.uleb128 0
.LLSDACSEC2888:
	.section	.text.unlikely
	.text
	.size	_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, .-_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.section	.text.unlikely
	.size	_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE.cold.181, .-_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE.cold.181
.LCOLDE2:
	.text
.LHOTE2:
	.section	.text.unlikely
	.align 2
.LCOLDB5:
	.text
.LHOTB5:
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel11_SolveNaiveEv
	.type	_ZN8KnnModel11_SolveNaiveEv, @function
_ZN8KnnModel11_SolveNaiveEv:
.LFB2889:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$576460752303423487, %rax
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movslq	(%rdi), %r14
	cmpq	%rax, %r14
	ja	.L196
	movq	%r14, %rbx
	salq	$4, %rbx
	movq	%rdi, %r12
	movq	%rbx, %rdi
	call	_Znam@PLT
	movq	%rax, %r13
	leaq	(%rax,%rbx), %rdx
	testq	%r14, %r14
	je	.L197
	.p2align 4,,10
	.p2align 3
.L200:
	movq	$0x000000000, (%rax)
	movl	$0, 8(%rax)
	addq	$16, %rax
	cmpq	%rax, %rdx
	jne	.L200
.L197:
	movslq	(%r12), %r10
	testl	%r10d, %r10d
	jle	.L217
	vmovsd	.LC4(%rip), %xmm3
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	vxorpd	%xmm4, %xmm4, %xmm4
	.p2align 4,,10
	.p2align 3
.L199:
	movl	%r15d, 8(%r13,%r14,2)
	vmovsd	%xmm3, 0(%r13,%r14,2)
	leaq	(%r14,%r14,2), %rbx
	xorl	%ecx, %ecx
	xorl	%edi, %edi
	.p2align 4,,10
	.p2align 3
.L206:
	cmpl	%edi, %r15d
	je	.L201
	movl	4(%r12), %esi
	testl	%esi, %esi
	jle	.L221
	movq	16(%r12), %rdx
	leaq	(%rcx,%rcx,2), %r9
	movq	(%rdx,%rbx), %r8
	movq	(%rdx,%r9), %r9
	leal	-1(%rsi), %edx
	cmpl	$2, %edx
	jbe	.L222
	movl	%esi, %r11d
	shrl	$2, %r11d
	salq	$5, %r11
	xorl	%edx, %edx
	vxorpd	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L204:
	vmovupd	(%r9,%rdx), %ymm2
	vfmadd231pd	(%r8,%rdx), %ymm2, %ymm0
	addq	$32, %rdx
	cmpq	%rdx, %r11
	jne	.L204
	vhaddpd	%ymm0, %ymm0, %ymm0
	movl	%esi, %edx
	andl	$-4, %edx
	vperm2f128	$1, %ymm0, %ymm0, %ymm1
	vaddpd	%ymm1, %ymm0, %ymm0
	cmpl	%edx, %esi
	je	.L205
.L203:
	movslq	%edx, %r11
	vmovsd	(%r8,%r11,8), %xmm5
	vfmadd231sd	(%r9,%r11,8), %xmm5, %xmm0
	leal	1(%rdx), %r11d
	cmpl	%esi, %r11d
	jge	.L205
	movslq	%r11d, %r11
	vmovsd	(%r9,%r11,8), %xmm6
	addl	$2, %edx
	vfmadd231sd	(%r8,%r11,8), %xmm6, %xmm0
	cmpl	%edx, %esi
	jle	.L205
	movslq	%edx, %rdx
	vmovsd	(%r9,%rdx,8), %xmm7
	vfmadd231sd	(%r8,%rdx,8), %xmm7, %xmm0
.L205:
	vaddsd	%xmm0, %xmm0, %xmm0
.L202:
	movq	72(%r12), %rdx
	vmovsd	(%rdx,%r14), %xmm1
	vaddsd	(%rdx,%rcx), %xmm1, %xmm1
	movl	%edi, 8(%r13,%rcx,2)
	vsubsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 0(%r13,%rcx,2)
.L201:
	incl	%edi
	addq	$8, %rcx
	cmpl	%r10d, %edi
	jl	.L206
	salq	$4, %r10
	movq	%r10, %rdx
	sarq	$4, %rdx
	movl	$63, %eax
	lzcntq	%rdx, %rdx
	subl	%edx, %eax
	leaq	0(%r13,%r10), %r8
	movslq	%eax, %rdx
	addq	%rdx, %rdx
	movq	%r8, %rsi
	movq	%r13, %rdi
	movq	%r10, 16(%rsp)
	movq	%r8, 24(%rsp)
	vzeroupper
	call	_ZSt16__introsort_loopIPSt4pairIdiElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_T1_.isra.144
	movq	16(%rsp), %r10
	movq	24(%rsp), %r8
	cmpq	$256, %r10
	jbe	.L207
	leaq	256(%r13), %r10
	movq	%r10, %rsi
	movq	%r13, %rdi
	call	_ZSt16__insertion_sortIPSt4pairIdiEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_.constprop.180
	movq	24(%rsp), %r8
	cmpq	%r10, %r8
	jne	.L208
	jmp	.L209
	.p2align 4,,10
	.p2align 3
.L232:
	vcomisd	%xmm0, %xmm1
	jne	.L211
	movl	-8(%rdx), %ecx
	cmpl	%ecx, %esi
	jl	.L213
.L211:
	addq	$16, %r10
	vmovsd	%xmm1, (%rdx)
	movl	%esi, 8(%rdx)
	cmpq	%r10, %r8
	je	.L209
.L208:
	vmovsd	(%r10), %xmm1
	movl	8(%r10), %esi
	movq	%r10, %rdx
.L215:
	vmovsd	-16(%rdx), %xmm0
	vcomisd	%xmm1, %xmm0
	jbe	.L232
	movl	-8(%rdx), %ecx
.L213:
	vmovsd	%xmm0, (%rdx)
	movl	%ecx, 8(%rdx)
	subq	$16, %rdx
	jmp	.L215
	.p2align 4,,10
	.p2align 3
.L221:
	vmovapd	%xmm4, %xmm0
	jmp	.L202
	.p2align 4,,10
	.p2align 3
.L222:
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%edx, %edx
	jmp	.L203
	.p2align 4,,10
	.p2align 3
.L207:
	movq	%r8, %rsi
	movq	%r13, %rdi
	call	_ZSt16__insertion_sortIPSt4pairIdiEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_.constprop.180
.L209:
	movl	8(%r12), %eax
	testl	%eax, %eax
	jle	.L218
	movq	48(%r12), %rdx
	movl	$1, %eax
	movq	(%rdx,%rbx), %rcx
	.p2align 4,,10
	.p2align 3
.L219:
	movq	%rax, %rdx
	salq	$4, %rdx
	movl	8(%r13,%rdx), %edx
	movl	%edx, -4(%rcx,%rax,4)
	incq	%rax
	cmpl	%eax, 8(%r12)
	jge	.L219
.L218:
	movslq	(%r12), %r10
	incl	%r15d
	addq	$8, %r14
	cmpl	%r15d, %r10d
	jg	.L199
.L217:
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	movq	%r13, %rdi
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	jmp	_ZdaPv@PLT
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	_ZN8KnnModel11_SolveNaiveEv.cold.182, @function
_ZN8KnnModel11_SolveNaiveEv.cold.182:
.LFSB2889:
.L196:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -56
	.cfi_offset 6, -16
	.cfi_offset 12, -48
	.cfi_offset 13, -40
	.cfi_offset 14, -32
	.cfi_offset 15, -24
	call	__cxa_throw_bad_array_new_length@PLT
	.cfi_endproc
.LFE2889:
	.text
	.size	_ZN8KnnModel11_SolveNaiveEv, .-_ZN8KnnModel11_SolveNaiveEv
	.section	.text.unlikely
	.size	_ZN8KnnModel11_SolveNaiveEv.cold.182, .-_ZN8KnnModel11_SolveNaiveEv.cold.182
.LCOLDE5:
	.text
.LHOTE5:
	.section	.text.unlikely
	.align 2
.LCOLDB6:
	.text
.LHOTB6:
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel12_SolveNaive2Ev
	.type	_ZN8KnnModel12_SolveNaive2Ev, @function
_ZN8KnnModel12_SolveNaive2Ev:
.LFB2907:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$1152921504606846975, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$64, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movslq	(%rdi), %rax
	cmpq	%rdx, %rax
	ja	.L234
	movq	%rdi, %r14
	leaq	0(,%rax,8), %rdi
	call	_Znam@PLT
	movslq	(%r14), %rbx
	movq	%rax, %r15
	testl	%ebx, %ebx
	jle	.L262
	xorl	%r12d, %r12d
	.p2align 4,,10
	.p2align 3
.L236:
	salq	$4, %rbx
	movq	%rbx, %rdi
	call	_Znam@PLT
	movq	%rax, %rdx
	leaq	(%rbx,%rax), %rdi
	.p2align 4,,10
	.p2align 3
.L237:
	movq	$0x000000000, (%rdx)
	movl	$0, 8(%rdx)
	addq	$16, %rdx
	cmpq	%rdx, %rdi
	jne	.L237
	movslq	(%r14), %rbx
	movq	%rax, (%r15,%r12,8)
	incq	%r12
	cmpl	%r12d, %ebx
	jg	.L236
	testl	%ebx, %ebx
	jle	.L262
	movq	$0, 48(%rsp)
	movq	$0, 24(%rsp)
	vmovsd	.LC4(%rip), %xmm3
	vxorpd	%xmm4, %xmm4, %xmm4
	.p2align 4,,10
	.p2align 3
.L257:
	movq	24(%rsp), %rax
	movq	(%r15,%rax,8), %rcx
	movq	%rax, %r12
	salq	$4, %r12
	movq	%rcx, 16(%rsp)
	addq	%r12, %rcx
	movl	%eax, 44(%rsp)
	leaq	0(,%rax,8), %r11
	movl	%eax, 8(%rcx)
	incl	%eax
	vmovsd	%xmm3, (%rcx)
	cmpl	%ebx, %eax
	jge	.L245
	movl	4(%r14), %r9d
	movq	72(%r14), %r13
	leal	-1(%r9), %edi
	movl	%r9d, %edx
	movl	%edi, 40(%rsp)
	shrl	$2, %edx
	movl	%r9d, %edi
	salq	$5, %rdx
	andl	$-4, %edi
	movq	%rdx, 56(%rsp)
	movl	%edi, 36(%rsp)
	addq	%r13, %r11
	cltq
	.p2align 4,,10
	.p2align 3
.L246:
	movl	%eax, %r10d
	testl	%r9d, %r9d
	jle	.L264
	movq	16(%r14), %rdx
	movq	48(%rsp), %rsi
	leaq	(%rax,%rax,2), %rdi
	cmpl	$2, 40(%rsp)
	movq	(%rdx,%rsi), %rsi
	movq	(%rdx,%rdi,8), %rdi
	jbe	.L265
	xorl	%edx, %edx
	vxorpd	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L243:
	vmovupd	(%rdi,%rdx), %ymm2
	vfmadd231pd	(%rsi,%rdx), %ymm2, %ymm0
	addq	$32, %rdx
	cmpq	%rdx, 56(%rsp)
	jne	.L243
	vhaddpd	%ymm0, %ymm0, %ymm0
	movl	36(%rsp), %edx
	vperm2f128	$1, %ymm0, %ymm0, %ymm1
	vaddpd	%ymm1, %ymm0, %ymm0
	cmpl	%r9d, %edx
	je	.L244
.L242:
	movslq	%edx, %r8
	vmovsd	(%rsi,%r8,8), %xmm5
	vfmadd231sd	(%rdi,%r8,8), %xmm5, %xmm0
	leal	1(%rdx), %r8d
	cmpl	%r9d, %r8d
	jge	.L244
	movslq	%r8d, %r8
	vmovsd	(%rdi,%r8,8), %xmm6
	addl	$2, %edx
	vfmadd231sd	(%rsi,%r8,8), %xmm6, %xmm0
	cmpl	%edx, %r9d
	jle	.L244
	movslq	%edx, %rdx
	vmovsd	(%rdi,%rdx,8), %xmm7
	vfmadd231sd	(%rsi,%rdx,8), %xmm7, %xmm0
.L244:
	vaddsd	%xmm0, %xmm0, %xmm0
.L241:
	vmovsd	(%r11), %xmm1
	movq	(%r15,%rax,8), %rdx
	vaddsd	0(%r13,%rax,8), %xmm1, %xmm1
	movl	44(%rsp), %esi
	addq	%r12, %rdx
	vsubsd	%xmm0, %xmm1, %xmm0
	incq	%rax
	movl	%r10d, 24(%rcx)
	vmovsd	%xmm0, 16(%rcx)
	vmovsd	%xmm0, (%rdx)
	movl	%esi, 8(%rdx)
	addq	$16, %rcx
	cmpl	%eax, %ebx
	jg	.L246
	vzeroupper
.L245:
	salq	$4, %rbx
	movq	%rbx, %rax
	sarq	$4, %rax
	movq	16(%rsp), %rdi
	movl	$63, %edx
	lzcntq	%rax, %rax
	subl	%eax, %edx
	leaq	(%rdi,%rbx), %r12
	movslq	%edx, %rdx
	addq	%rdx, %rdx
	movq	%r12, %rsi
	call	_ZSt16__introsort_loopIPSt4pairIdiElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_T1_.isra.144
	cmpq	$256, %rbx
	jbe	.L274
	movq	16(%rsp), %rdi
	leaq	256(%rdi), %r10
	movq	%r10, %rsi
	call	_ZSt16__insertion_sortIPSt4pairIdiEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_.constprop.180
	cmpq	%r10, %r12
	jne	.L247
	jmp	.L248
	.p2align 4,,10
	.p2align 3
.L275:
	vcomisd	%xmm0, %xmm1
	jne	.L250
	movl	-8(%rax), %edx
	cmpl	%edx, %ecx
	jl	.L252
.L250:
	addq	$16, %r10
	vmovsd	%xmm1, (%rax)
	movl	%ecx, 8(%rax)
	cmpq	%r10, %r12
	je	.L248
.L247:
	vmovsd	(%r10), %xmm1
	movl	8(%r10), %ecx
	movq	%r10, %rax
.L254:
	vmovsd	-16(%rax), %xmm0
	vcomisd	%xmm1, %xmm0
	jbe	.L275
	movl	-8(%rax), %edx
.L252:
	vmovsd	%xmm0, (%rax)
	movl	%edx, 8(%rax)
	subq	$16, %rax
	jmp	.L254
	.p2align 4,,10
	.p2align 3
.L274:
	movq	16(%rsp), %rdi
	movq	%r12, %rsi
	call	_ZSt16__insertion_sortIPSt4pairIdiEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S6_T0_.constprop.180
.L248:
	movl	8(%r14), %eax
	testl	%eax, %eax
	jle	.L258
	movq	24(%rsp), %rax
	movq	48(%rsp), %rbx
	movq	(%r15,%rax,8), %rsi
	movq	48(%r14), %rax
	movq	(%rax,%rbx), %rcx
	movl	$1, %eax
	.p2align 4,,10
	.p2align 3
.L259:
	movq	%rax, %rdx
	salq	$4, %rdx
	movl	8(%rsi,%rdx), %edx
	movl	%edx, -4(%rcx,%rax,4)
	incq	%rax
	cmpl	%eax, 8(%r14)
	jge	.L259
.L258:
	incq	24(%rsp)
	movslq	(%r14), %rbx
	addq	$24, 48(%rsp)
	movq	24(%rsp), %rax
	cmpl	%eax, %ebx
	jg	.L257
	testl	%ebx, %ebx
	jle	.L262
	xorl	%ebx, %ebx
	.p2align 4,,10
	.p2align 3
.L260:
	movq	(%r15,%rbx,8), %rdi
	testq	%rdi, %rdi
	je	.L261
	incq	%rbx
	call	_ZdaPv@PLT
	cmpl	%ebx, (%r14)
	jg	.L260
.L262:
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	movq	%r15, %rdi
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	jmp	_ZdaPv@PLT
	.p2align 4,,10
	.p2align 3
.L264:
	.cfi_restore_state
	vmovapd	%xmm4, %xmm0
	jmp	.L241
	.p2align 4,,10
	.p2align 3
.L265:
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%edx, %edx
	jmp	.L242
	.p2align 4,,10
	.p2align 3
.L261:
	incq	%rbx
	cmpl	%ebx, (%r14)
	jg	.L260
	jmp	.L262
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	_ZN8KnnModel12_SolveNaive2Ev.cold.183, @function
_ZN8KnnModel12_SolveNaive2Ev.cold.183:
.LFSB2907:
.L234:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -56
	.cfi_offset 6, -16
	.cfi_offset 12, -48
	.cfi_offset 13, -40
	.cfi_offset 14, -32
	.cfi_offset 15, -24
	call	__cxa_throw_bad_array_new_length@PLT
	.cfi_endproc
.LFE2907:
	.text
	.size	_ZN8KnnModel12_SolveNaive2Ev, .-_ZN8KnnModel12_SolveNaive2Ev
	.section	.text.unlikely
	.size	_ZN8KnnModel12_SolveNaive2Ev.cold.183, .-_ZN8KnnModel12_SolveNaive2Ev.cold.183
.LCOLDE6:
	.text
.LHOTE6:
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel8DistanceEii
	.type	_ZN8KnnModel8DistanceEii, @function
_ZN8KnnModel8DistanceEii:
.LFB2924:
	.cfi_startproc
	movl	4(%rdi), %r10d
	movslq	%esi, %rsi
	movslq	%edx, %rdx
	testl	%r10d, %r10d
	jle	.L281
	movq	16(%rdi), %rax
	leaq	(%rsi,%rsi,2), %rcx
	movq	(%rax,%rcx,8), %r8
	leaq	(%rdx,%rdx,2), %rcx
	movq	(%rax,%rcx,8), %r9
	leal	-1(%r10), %eax
	cmpl	$2, %eax
	jbe	.L282
	movl	%r10d, %ecx
	shrl	$2, %ecx
	salq	$5, %rcx
	xorl	%eax, %eax
	vxorpd	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L279:
	vmovupd	(%r9,%rax), %ymm2
	vfmadd231pd	(%r8,%rax), %ymm2, %ymm0
	addq	$32, %rax
	cmpq	%rcx, %rax
	jne	.L279
	vhaddpd	%ymm0, %ymm0, %ymm0
	movl	%r10d, %eax
	andl	$-4, %eax
	vperm2f128	$1, %ymm0, %ymm0, %ymm1
	vaddpd	%ymm1, %ymm0, %ymm0
	cmpl	%r10d, %eax
	je	.L285
	vzeroupper
.L278:
	movslq	%eax, %rcx
	vmovsd	(%r9,%rcx,8), %xmm3
	vfmadd231sd	(%r8,%rcx,8), %xmm3, %xmm0
	leal	1(%rax), %ecx
	cmpl	%r10d, %ecx
	jge	.L280
	movslq	%ecx, %rcx
	vmovsd	(%r8,%rcx,8), %xmm4
	addl	$2, %eax
	vfmadd231sd	(%r9,%rcx,8), %xmm4, %xmm0
	cmpl	%eax, %r10d
	jle	.L280
	cltq
	vmovsd	(%r9,%rax,8), %xmm5
	vfmadd231sd	(%r8,%rax,8), %xmm5, %xmm0
.L280:
	vaddsd	%xmm0, %xmm0, %xmm0
.L277:
	movq	72(%rdi), %rax
	vmovsd	(%rax,%rsi,8), %xmm1
	vaddsd	(%rax,%rdx,8), %xmm1, %xmm1
	vsubsd	%xmm0, %xmm1, %xmm0
	ret
	.p2align 4,,10
	.p2align 3
.L285:
	vzeroupper
	jmp	.L280
	.p2align 4,,10
	.p2align 3
.L281:
	vxorpd	%xmm0, %xmm0, %xmm0
	jmp	.L277
.L282:
	xorl	%eax, %eax
	vxorpd	%xmm0, %xmm0, %xmm0
	jmp	.L278
	.cfi_endproc
.LFE2924:
	.size	_ZN8KnnModel8DistanceEii, .-_ZN8KnnModel8DistanceEii
	.section	.text._Z8pop_heapISt4pairIdiEEvPT_S3_,"axG",@progbits,_Z8pop_heapISt4pairIdiEEvPT_S3_,comdat
	.p2align 4,,15
	.weak	_Z8pop_heapISt4pairIdiEEvPT_S3_
	.type	_Z8pop_heapISt4pairIdiEEvPT_S3_, @function
_Z8pop_heapISt4pairIdiEEvPT_S3_:
.LFB3267:
	.cfi_startproc
	vmovsd	(%rdi), %xmm0
	vmovsd	-16(%rsi), %xmm1
	movl	8(%rdi), %eax
	movl	-8(%rsi), %edx
	vmovsd	%xmm1, (%rdi)
	leaq	-16(%rsi), %r8
	vmovsd	%xmm0, -16(%rsi)
	movl	%edx, 8(%rdi)
	movl	%eax, -8(%rsi)
	leaq	16(%rdi), %rax
	leaq	32(%rdi), %rdx
	cmpq	%rax, %r8
	jbe	.L302
	vmovsd	(%rdi), %xmm1
	movl	$2, %r9d
	movl	$1, %ecx
.L295:
	vmovsd	(%rax), %xmm0
	cmpq	%rdx, %r8
	jbe	.L288
	vmovsd	(%rdx), %xmm2
	vcomisd	%xmm0, %xmm2
	ja	.L297
	je	.L303
	.p2align 4,,10
	.p2align 3
.L288:
	vcomisd	%xmm1, %xmm0
	jbe	.L301
	movl	8(%rdi), %edx
	movl	8(%rax), %esi
.L292:
	leal	1(%rcx,%rcx), %r10d
	vmovsd	%xmm0, (%rdi)
	leal	2(%rcx,%rcx), %r9d
	vmovsd	%xmm1, (%rax)
	movl	%esi, 8(%rdi)
	movzwl	%cx, %esi
	movzwl	%r10w, %ecx
	movl	%edx, 8(%rax)
	subq	%rsi, %rcx
	movzwl	%r9w, %edx
	salq	$4, %rcx
	subq	%rsi, %rdx
	salq	$4, %rdx
	addq	%rax, %rcx
	addq	%rax, %rdx
	movq	%rax, %rdi
	cmpq	%rcx, %r8
	jbe	.L302
	movq	%rcx, %rax
	movl	%r10d, %ecx
	jmp	.L295
	.p2align 4,,10
	.p2align 3
.L301:
	vcomisd	%xmm0, %xmm1
	jne	.L302
	movl	8(%rdi), %edx
	movl	8(%rax), %esi
	cmpl	%esi, %edx
	jl	.L292
.L302:
	ret
	.p2align 4,,10
	.p2align 3
.L303:
	movl	8(%rdx), %esi
	cmpl	%esi, 8(%rax)
	jge	.L288
.L297:
	vmovapd	%xmm2, %xmm0
	movq	%rdx, %rax
	movl	%r9d, %ecx
	jmp	.L288
	.cfi_endproc
.LFE3267:
	.size	_Z8pop_heapISt4pairIdiEEvPT_S3_, .-_Z8pop_heapISt4pairIdiEEvPT_S3_
	.section	.text.unlikely
	.align 2
.LCOLDB7:
	.text
.LHOTB7:
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel10_SolveHeapEv
	.type	_ZN8KnnModel10_SolveHeapEv, @function
_ZN8KnnModel10_SolveHeapEv:
.LFB2916:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$1152921504606846975, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$64, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movslq	(%rdi), %rax
	cmpq	%rdx, %rax
	ja	.L305
	movq	%rdi, %r12
	leaq	0(,%rax,8), %rdi
	call	_Znam@PLT
	movq	%rax, %r13
	movl	(%r12), %eax
	testl	%eax, %eax
	jle	.L345
	movslq	8(%r12), %r15
	movabsq	$576460752303423487, %rax
	cmpq	%rax, %r15
	ja	.L376
	movq	%rax, 56(%rsp)
	movl	$1, %ebx
	.p2align 4,,10
	.p2align 3
.L308:
	movq	%r15, %r14
	salq	$4, %r14
	movq	%r14, %rdi
	call	_Znam@PLT
	movq	%rax, %rdx
	leaq	(%r14,%rax), %rdi
	testq	%r15, %r15
	je	.L313
	.p2align 4,,10
	.p2align 3
.L314:
	movq	$0x000000000, (%rdx)
	movl	$0, 8(%rdx)
	addq	$16, %rdx
	cmpq	%rdx, %rdi
	jne	.L314
.L313:
	movl	(%r12), %edx
	movq	%rax, -8(%r13,%rbx,8)
	cmpl	%ebx, %edx
	jle	.L377
	movslq	8(%r12), %r15
	incq	%rbx
	cmpq	56(%rsp), %r15
	jbe	.L308
	jmp	.L305
.L371:
	vzeroupper
.L345:
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	movq	%r13, %rdi
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	jmp	_ZdaPv@PLT
.L377:
	.cfi_restore_state
	testl	%edx, %edx
	jle	.L345
	movq	$0, 56(%rsp)
	movslq	8(%r12), %rax
	xorl	%r14d, %r14d
	.p2align 4,,10
	.p2align 3
.L344:
	leaq	0(,%r14,8), %rbx
	leal	1(%r14), %r9d
	movl	%r14d, %r8d
	movq	%rbx, 48(%rsp)
	movq	0(%r13,%r14,8), %rcx
	cmpl	%edx, %r9d
	jge	.L315
	movq	%r14, %rbx
	movslq	%r9d, %r15
	movq	%r13, %r14
	movl	%r8d, %r13d
	movq	%rbx, %r8
	.p2align 4,,10
	.p2align 3
.L338:
	movl	4(%r12), %esi
	movslq	%r15d, %rbx
	testl	%esi, %esi
	jle	.L350
	movq	16(%r12), %rdx
	movq	56(%rsp), %rdi
	movq	(%rdx,%rdi), %r10
	leaq	(%r15,%r15,2), %rdi
	movq	(%rdx,%rdi,8), %rdi
	leal	-1(%rsi), %edx
	cmpl	$2, %edx
	jbe	.L351
	movl	%esi, %r11d
	shrl	$2, %r11d
	salq	$5, %r11
	xorl	%edx, %edx
	vxorpd	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L318:
	vmovupd	(%rdi,%rdx), %ymm3
	vfmadd231pd	(%r10,%rdx), %ymm3, %ymm0
	addq	$32, %rdx
	cmpq	%rdx, %r11
	jne	.L318
	vhaddpd	%ymm0, %ymm0, %ymm0
	movl	%esi, %edx
	andl	$-4, %edx
	vperm2f128	$1, %ymm0, %ymm0, %ymm1
	vaddpd	%ymm1, %ymm0, %ymm0
	cmpl	%esi, %edx
	je	.L319
.L317:
	movslq	%edx, %r11
	vmovsd	(%r10,%r11,8), %xmm4
	vfmadd231sd	(%rdi,%r11,8), %xmm4, %xmm0
	leal	1(%rdx), %r11d
	cmpl	%esi, %r11d
	jge	.L319
	movslq	%r11d, %r11
	vmovsd	(%rdi,%r11,8), %xmm5
	addl	$2, %edx
	vfmadd231sd	(%r10,%r11,8), %xmm5, %xmm0
	cmpl	%edx, %esi
	jle	.L319
	movslq	%edx, %rdx
	vmovsd	(%rdi,%rdx,8), %xmm6
	vfmadd231sd	(%r10,%rdx,8), %xmm6, %xmm0
.L319:
	vaddsd	%xmm0, %xmm0, %xmm0
.L316:
	movq	72(%r12), %rdx
	movq	48(%rsp), %rdi
	vmovsd	(%rdx,%r15,8), %xmm1
	vaddsd	(%rdx,%rdi), %xmm1, %xmm1
	vsubsd	%xmm0, %xmm1, %xmm0
	cmpl	%r15d, %eax
	jge	.L320
	vcomisd	(%rcx), %xmm0
	jnb	.L321
	movslq	%eax, %rsi
	salq	$4, %rsi
	addq	%rcx, %rsi
	movq	%rcx, %rdi
	movq	%r8, 24(%rsp)
	movl	%r9d, 36(%rsp)
	vmovsd	%xmm0, 40(%rsp)
	vzeroupper
	call	_Z8pop_heapISt4pairIdiEEvPT_S3_
	movq	24(%rsp), %r8
	movslq	8(%r12), %rax
	movq	(%r14,%r8,8), %rcx
	movl	36(%rsp), %r9d
	vmovsd	40(%rsp), %xmm0
.L320:
	cmpl	%eax, %ebx
	cmovle	%rbx, %rax
	salq	$4, %rax
	leaq	-16(%rcx,%rax), %rax
	movl	%ebx, 8(%rax)
	vmovsd	%xmm0, (%rax)
	movslq	8(%r12), %rax
	cmpl	%ebx, %eax
	cmovle	%eax, %ebx
	movslq	%ebx, %rdx
	salq	$4, %rdx
	subq	$16, %rdx
	leaq	(%rcx,%rdx), %rsi
	sarq	$4, %rdx
	cmpq	%rcx, %rsi
	je	.L321
.L327:
	leal	-1(%rdx), %edi
	movzwl	%di, %edi
	sarl	%edi
	movslq	%edi, %rax
	movzwl	%dx, %edx
	subq	%rdx, %rax
	salq	$4, %rax
	addq	%rsi, %rax
	vmovsd	(%rax), %xmm1
	vmovsd	(%rsi), %xmm2
	vcomisd	%xmm1, %xmm2
	jbe	.L367
	movl	8(%rax), %edx
	movl	8(%rsi), %r10d
.L325:
	vmovsd	%xmm2, (%rax)
	vmovsd	%xmm1, (%rsi)
	movl	%r10d, 8(%rax)
	movl	%edx, 8(%rsi)
	movq	%rax, %rsi
	movl	%edi, %edx
	cmpq	%rcx, %rax
	jne	.L327
	.p2align 4,,10
	.p2align 3
.L373:
	movslq	8(%r12), %rax
.L321:
	movq	(%r14,%r15,8), %rdi
	cmpl	%r8d, %eax
	jg	.L328
	vcomisd	(%rdi), %xmm0
	vmovsd	%xmm0, 40(%rsp)
	jnb	.L329
	movslq	%eax, %rsi
	salq	$4, %rsi
	addq	%rdi, %rsi
	movq	%r8, 24(%rsp)
	movl	%r9d, 36(%rsp)
	vzeroupper
	call	_Z8pop_heapISt4pairIdiEEvPT_S3_
	movq	24(%rsp), %r8
	movq	(%r14,%r15,8), %rdi
	movl	8(%r12), %eax
	movq	(%r14,%r8,8), %rcx
	vmovsd	40(%rsp), %xmm0
	movl	36(%rsp), %r9d
.L328:
	decl	%eax
	cmpl	%r13d, %eax
	cmovg	%r13d, %eax
	movslq	%r9d, %r10
	cltq
	salq	$4, %rax
	addq	%rdi, %rax
	movl	%r13d, 8(%rax)
	vmovsd	%xmm0, (%rax)
	movslq	8(%r12), %rax
	cmpl	%r9d, %eax
	cmovle	%rax, %r10
	salq	$4, %r10
	subq	$16, %r10
	leaq	(%rdi,%r10), %rsi
	sarq	$4, %r10
	cmpq	%rdi, %rsi
	je	.L329
.L337:
	leal	-1(%r10), %edx
	movzwl	%dx, %edx
	sarl	%edx
	movslq	%edx, %rax
	movzwl	%r10w, %r10d
	subq	%r10, %rax
	salq	$4, %rax
	addq	%rsi, %rax
	vmovsd	(%rax), %xmm0
	vmovsd	(%rsi), %xmm1
	vcomisd	%xmm0, %xmm1
	jbe	.L368
	movl	8(%rax), %r10d
	movl	8(%rsi), %r11d
.L334:
	vmovsd	%xmm1, (%rax)
	vmovsd	%xmm0, (%rsi)
	movl	%r11d, 8(%rax)
	movl	%r10d, 8(%rsi)
	movq	%rax, %rsi
	movl	%edx, %r10d
	cmpq	%rdi, %rax
	jne	.L337
	.p2align 4,,10
	.p2align 3
.L374:
	movslq	8(%r12), %rax
.L329:
	incq	%r15
	cmpl	%r15d, (%r12)
	jg	.L338
	movq	%r14, %r13
	movq	%r8, %r14
.L315:
	movslq	%eax, %rbx
	salq	$4, %rbx
	addq	%rcx, %rbx
	cmpq	%rcx, %rbx
	je	.L342
	movq	%rcx, %r15
	vzeroupper
	.p2align 4,,10
	.p2align 3
.L339:
	movq	%rbx, %rsi
	movq	%r15, %rdi
	subq	$16, %rbx
	call	_Z8pop_heapISt4pairIdiEEvPT_S3_
	cmpq	%r15, %rbx
	jne	.L339
	movslq	8(%r12), %rax
.L342:
	testl	%eax, %eax
	jle	.L340
	movq	48(%r12), %rax
	movq	56(%rsp), %rbx
	movq	0(%r13,%r14,8), %rsi
	movq	(%rax,%rbx), %rcx
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L343:
	movq	%rdx, %rax
	salq	$4, %rax
	movl	8(%rsi,%rax), %eax
	movl	%eax, (%rcx,%rdx,4)
	incq	%rdx
	movslq	8(%r12), %rax
	cmpl	%edx, %eax
	jg	.L343
.L340:
	movl	(%r12), %edx
	incq	%r14
	addq	$24, 56(%rsp)
	cmpl	%r14d, %edx
	jg	.L344
	testl	%edx, %edx
	jle	.L371
	xorl	%ebx, %ebx
	vzeroupper
.L348:
	movq	0(%r13,%rbx,8), %rdi
	testq	%rdi, %rdi
	je	.L346
.L378:
	call	_ZdaPv@PLT
	movl	(%r12), %edx
	incq	%rbx
	cmpl	%ebx, %edx
	jle	.L345
	movq	0(%r13,%rbx,8), %rdi
	testq	%rdi, %rdi
	jne	.L378
.L346:
	incq	%rbx
	cmpl	%ebx, %edx
	jg	.L348
	jmp	.L345
	.p2align 4,,10
	.p2align 3
.L367:
	vcomisd	%xmm2, %xmm1
	jne	.L373
	movl	8(%rax), %edx
	movl	8(%rsi), %r10d
	cmpl	%r10d, %edx
	jl	.L325
	jmp	.L373
	.p2align 4,,10
	.p2align 3
.L368:
	vcomisd	%xmm1, %xmm0
	jne	.L374
	movl	8(%rax), %r10d
	movl	8(%rsi), %r11d
	cmpl	%r11d, %r10d
	jge	.L374
	jmp	.L334
	.p2align 4,,10
	.p2align 3
.L350:
	vxorpd	%xmm0, %xmm0, %xmm0
	jmp	.L316
	.p2align 4,,10
	.p2align 3
.L351:
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%edx, %edx
	jmp	.L317
.L376:
	jmp	.L305
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	_ZN8KnnModel10_SolveHeapEv.cold.184, @function
_ZN8KnnModel10_SolveHeapEv.cold.184:
.LFSB2916:
.L305:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -56
	.cfi_offset 6, -16
	.cfi_offset 12, -48
	.cfi_offset 13, -40
	.cfi_offset 14, -32
	.cfi_offset 15, -24
	call	__cxa_throw_bad_array_new_length@PLT
	.cfi_endproc
.LFE2916:
	.text
	.size	_ZN8KnnModel10_SolveHeapEv, .-_ZN8KnnModel10_SolveHeapEv
	.section	.text.unlikely
	.size	_ZN8KnnModel10_SolveHeapEv.cold.184, .-_ZN8KnnModel10_SolveHeapEv.cold.184
.LCOLDE7:
	.text
.LHOTE7:
	.section	.rodata._ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd.str1.1,"aMS",@progbits,1
.LC8:
	.string	"vector::_M_fill_insert"
	.section	.text._ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd,"axG",@progbits,_ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd
	.type	_ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd, @function
_ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd:
.LFB3490:
	.cfi_startproc
	testq	%rdx, %rdx
	je	.L443
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	.cfi_offset 15, -24
	movq	%rsi, %r15
	pushq	%r14
	.cfi_offset 14, -32
	movq	%rdi, %r14
	pushq	%r13
	.cfi_offset 13, -40
	movq	%rsi, %r13
	pushq	%r12
	pushq	%rbx
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdx, %rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	movq	8(%rdi), %r12
	movq	16(%rdi), %rax
	subq	%r12, %rax
	sarq	$3, %rax
	cmpq	%rdx, %rax
	jb	.L382
	movq	%r12, %r8
	subq	%rsi, %r8
	movq	%r8, %rax
	sarq	$3, %rax
	vmovsd	(%rcx), %xmm0
	cmpq	%rax, %rdx
	jnb	.L383
	salq	$3, %rbx
	movq	%r12, %rcx
	subq	%rbx, %rcx
	movq	%r12, %rax
	cmpq	%rcx, %r12
	je	.L384
	movq	%rcx, %rsi
	movq	%rbx, %rdx
	movq	%r12, %rdi
	vmovsd	%xmm0, 16(%rsp)
	movq	%rcx, 24(%rsp)
	call	memmove@PLT
	movq	8(%r14), %rax
	vmovsd	16(%rsp), %xmm0
	movq	24(%rsp), %rcx
.L384:
	addq	%rbx, %rax
	movq	%rax, 8(%r14)
	cmpq	%rcx, %r13
	je	.L385
	movq	%rcx, %rdx
	subq	%r13, %rdx
	movq	%r12, %rdi
	subq	%rdx, %rdi
	movq	%r13, %rsi
	vmovsd	%xmm0, 24(%rsp)
	call	memmove@PLT
	vmovsd	24(%rsp), %xmm0
.L385:
	addq	%r13, %rbx
	cmpq	%rbx, %r13
	je	.L441
	movq	%rbx, %rdx
	subq	%r13, %rdx
	leaq	-8(%rdx), %rcx
	movq	%rcx, %rdx
	shrq	$3, %rdx
	movq	%r13, %rax
	incq	%rdx
	cmpq	$16, %rcx
	jbe	.L387
	movq	%rdx, %rcx
	shrq	$2, %rcx
	salq	$5, %rcx
	vbroadcastsd	%xmm0, %ymm1
	addq	%r13, %rcx
	.p2align 4,,10
	.p2align 3
.L389:
	vmovupd	%ymm1, (%rax)
	addq	$32, %rax
	cmpq	%rax, %rcx
	jne	.L389
	movq	%rdx, %rax
	andq	$-4, %rax
	leaq	0(%r13,%rax,8), %r15
	cmpq	%rax, %rdx
	je	.L439
	vzeroupper
.L387:
	leaq	8(%r15), %rax
	vmovsd	%xmm0, (%r15)
	cmpq	%rax, %rbx
	je	.L441
	leaq	16(%r15), %rax
	vmovsd	%xmm0, 8(%r15)
	cmpq	%rax, %rbx
	je	.L441
.L446:
	vmovsd	%xmm0, 16(%r15)
.L441:
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L443:
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	.cfi_restore 15
	ret
	.p2align 4,,10
	.p2align 3
.L383:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -56
	.cfi_offset 6, -16
	.cfi_offset 12, -48
	.cfi_offset 13, -40
	.cfi_offset 14, -32
	.cfi_offset 15, -24
	subq	%rax, %rbx
	je	.L414
	leaq	-1(%rbx), %rax
	cmpq	$2, %rax
	jbe	.L415
	movq	%rbx, %rdx
	shrq	$2, %rdx
	vbroadcastsd	%xmm0, %ymm1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L393:
	movq	%rax, %rcx
	salq	$5, %rcx
	incq	%rax
	vmovupd	%ymm1, (%r12,%rcx)
	cmpq	%rax, %rdx
	jne	.L393
	movq	%rbx, %rax
	andq	$-4, %rax
	movq	%rbx, %rcx
	subq	%rax, %rcx
	leaq	(%r12,%rax,8), %rdx
	cmpq	%rax, %rbx
	je	.L447
	vzeroupper
.L392:
	vmovsd	%xmm0, (%rdx)
	cmpq	$1, %rcx
	je	.L394
	vmovsd	%xmm0, 8(%rdx)
	cmpq	$2, %rcx
	je	.L394
	vmovsd	%xmm0, 16(%rdx)
.L394:
	leaq	(%r12,%rbx,8), %rdi
.L391:
	vmovsd	%xmm0, 24(%rsp)
	movq	%rdi, 8(%r14)
	cmpq	%r13, %r12
	je	.L395
	movq	%r8, %rdx
	movq	%r13, %rsi
	movq	%r8, 16(%rsp)
	call	memmove@PLT
	movq	%r12, %rdx
	subq	%r13, %rdx
	subq	$8, %rdx
	movq	16(%rsp), %r8
	movq	%rdx, %rcx
	shrq	$3, %rcx
	addq	%r8, 8(%r14)
	incq	%rcx
	cmpq	$16, %rdx
	movq	%r13, %rax
	vmovsd	24(%rsp), %xmm0
	jbe	.L396
	movq	%rcx, %rdx
	shrq	$2, %rdx
	salq	$5, %rdx
	vbroadcastsd	%xmm0, %ymm1
	addq	%r13, %rdx
	.p2align 4,,10
	.p2align 3
.L398:
	vmovupd	%ymm1, (%rax)
	addq	$32, %rax
	cmpq	%rdx, %rax
	jne	.L398
	movq	%rcx, %rax
	andq	$-4, %rax
	leaq	0(%r13,%rax,8), %r15
	cmpq	%rcx, %rax
	je	.L439
	vzeroupper
.L396:
	leaq	8(%r15), %rax
	vmovsd	%xmm0, (%r15)
	cmpq	%rax, %r12
	je	.L441
	leaq	16(%r15), %rax
	vmovsd	%xmm0, 8(%r15)
	cmpq	%rax, %r12
	jne	.L446
	jmp	.L441
	.p2align 4,,10
	.p2align 3
.L382:
	movq	(%rdi), %r15
	movabsq	$2305843009213693951, %rdx
	subq	%r15, %r12
	sarq	$3, %r12
	movq	%rdx, %rax
	subq	%r12, %rax
	cmpq	%rax, %rbx
	ja	.L448
	cmpq	%r12, %rbx
	movq	%r12, %rax
	cmovnb	%rbx, %rax
	addq	%rax, %r12
	setc	%al
	movzbl	%al, %eax
	subq	%r15, %rsi
	testq	%rax, %rax
	jne	.L417
	cmpq	%rdx, %r12
	ja	.L417
	testq	%r12, %r12
	jne	.L449
	movq	%rsi, %rdx
	xorl	%r12d, %r12d
	xorl	%r8d, %r8d
	jmp	.L404
	.p2align 4,,10
	.p2align 3
.L417:
	movq	$-8, %r12
.L403:
	movq	%r12, %rdi
	movq	%rcx, 16(%rsp)
	movq	%rsi, 24(%rsp)
	call	_Znwm@PLT
	movq	(%r14), %r15
	movq	%r13, %rdx
	movq	24(%rsp), %rsi
	movq	16(%rsp), %rcx
	movq	%rax, %r8
	addq	%rax, %r12
	subq	%r15, %rdx
.L404:
	leaq	-1(%rbx), %rax
	addq	%r8, %rsi
	vmovsd	(%rcx), %xmm1
	cmpq	$2, %rax
	jbe	.L419
	leaq	-4(%rbx), %rcx
	shrq	$2, %rcx
	incq	%rcx
	vbroadcastsd	%xmm1, %ymm0
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L406:
	movq	%rax, %rdi
	salq	$5, %rdi
	incq	%rax
	vmovupd	%ymm0, (%rsi,%rdi)
	cmpq	%rax, %rcx
	ja	.L406
	leaq	0(,%rcx,4), %rax
	movq	%rbx, %rdi
	salq	$5, %rcx
	subq	%rax, %rdi
	addq	%rcx, %rsi
	cmpq	%rax, %rbx
	je	.L450
	vzeroupper
.L405:
	vmovsd	%xmm1, (%rsi)
	cmpq	$1, %rdi
	je	.L407
	vmovsd	%xmm1, 8(%rsi)
	cmpq	$2, %rdi
	je	.L407
	vmovsd	%xmm1, 16(%rsi)
.L407:
	leaq	(%rdx,%rbx,8), %rcx
	addq	%r8, %rcx
	cmpq	%r15, %r13
	je	.L408
	movq	%r8, %rdi
	movq	%r15, %rsi
	movq	%rcx, 24(%rsp)
	call	memmove@PLT
	movq	%rax, %r8
	movq	8(%r14), %rax
	movq	24(%rsp), %rcx
	movq	%rax, %rdx
	subq	%r13, %rdx
	leaq	(%rcx,%rdx), %rbx
	cmpq	%rax, %r13
	je	.L410
.L409:
	movq	%r13, %rsi
	movq	%rcx, %rdi
	movq	%r8, 24(%rsp)
	call	memcpy@PLT
	movq	24(%rsp), %r8
.L411:
	testq	%r15, %r15
	jne	.L410
.L412:
	movq	%r8, (%r14)
	movq	%rbx, 8(%r14)
	movq	%r12, 16(%r14)
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L410:
	.cfi_restore_state
	movq	%r15, %rdi
	movq	%r8, 24(%rsp)
	call	_ZdlPv@PLT
	movq	24(%rsp), %r8
	jmp	.L412
	.p2align 4,,10
	.p2align 3
.L408:
	movq	8(%r14), %rax
	movq	%rax, %rdx
	subq	%r13, %rdx
	leaq	(%rcx,%rdx), %rbx
	cmpq	%rax, %r13
	jne	.L409
	jmp	.L411
	.p2align 4,,10
	.p2align 3
.L439:
	vzeroupper
	jmp	.L441
	.p2align 4,,10
	.p2align 3
.L449:
	salq	$3, %r12
	jmp	.L403
	.p2align 4,,10
	.p2align 3
.L450:
	vzeroupper
	jmp	.L407
	.p2align 4,,10
	.p2align 3
.L447:
	vzeroupper
	jmp	.L394
	.p2align 4,,10
	.p2align 3
.L414:
	movq	%r12, %rdi
	jmp	.L391
	.p2align 4,,10
	.p2align 3
.L395:
	addq	%r8, %rdi
	movq	%rdi, 8(%r14)
	jmp	.L441
	.p2align 4,,10
	.p2align 3
.L419:
	movq	%rbx, %rdi
	jmp	.L405
.L415:
	movq	%r12, %rdx
	movq	%rbx, %rcx
	jmp	.L392
.L448:
	leaq	.LC8(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE3490:
	.size	_ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd, .-_ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd
	.text
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel24PreCalculationOfDistanceEv
	.type	_ZN8KnnModel24PreCalculationOfDistanceEv, @function
_ZN8KnnModel24PreCalculationOfDistanceEv:
.LFB2925:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	subq	$16, %rsp
	.cfi_def_cfa_offset 32
	movq	80(%rbx), %rsi
	movslq	(%rdi), %rdx
	movq	72(%rdi), %rdi
	movq	%fs:40, %rax
	movq	%rax, 8(%rsp)
	xorl	%eax, %eax
	movq	%rsi, %rax
	subq	%rdi, %rax
	sarq	$3, %rax
	movq	$0x000000000, (%rsp)
	cmpq	%rax, %rdx
	ja	.L463
	movq	%rdx, %rcx
	jb	.L464
.L453:
	testl	%ecx, %ecx
	jle	.L451
	movl	4(%rbx), %eax
	testl	%eax, %eax
	jle	.L451
	leal	-1(%rcx), %edx
	decl	%eax
	movq	16(%rbx), %r9
	movq	72(%rbx), %r8
	leaq	8(,%rdx,8), %r10
	xorl	%esi, %esi
	leaq	8(,%rax,8), %rdi
	.p2align 4,,10
	.p2align 3
.L458:
	leaq	(%rsi,%rsi,2), %rax
	leaq	(%r8,%rsi), %rdx
	movq	(%r9,%rax), %rax
	vmovsd	(%rdx), %xmm0
	leaq	(%rdi,%rax), %rcx
	.p2align 4,,10
	.p2align 3
.L456:
	vmovsd	(%rax), %xmm1
	addq	$8, %rax
	vfmadd231sd	%xmm1, %xmm1, %xmm0
	vmovsd	%xmm0, (%rdx)
	cmpq	%rax, %rcx
	jne	.L456
	addq	$8, %rsi
	cmpq	%r10, %rsi
	jne	.L458
.L451:
	movq	8(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L465
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
.L464:
	.cfi_restore_state
	leaq	(%rdi,%rdx,8), %rax
	cmpq	%rax, %rsi
	je	.L453
	movq	%rax, 80(%rbx)
	jmp	.L453
.L463:
	movq	%rsp, %rcx
	subq	%rax, %rdx
	leaq	72(%rbx), %rdi
	call	_ZNSt6vectorIdSaIdEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPdS1_EEmRKd
	movl	(%rbx), %ecx
	jmp	.L453
.L465:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE2925:
	.size	_ZN8KnnModel24PreCalculationOfDistanceEv, .-_ZN8KnnModel24PreCalculationOfDistanceEv
	.section	.text._ZNSt6vectorIdSaIdEEaSERKS1_,"axG",@progbits,_ZNSt6vectorIdSaIdEEaSERKS1_,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZNSt6vectorIdSaIdEEaSERKS1_
	.type	_ZNSt6vectorIdSaIdEEaSERKS1_, @function
_ZNSt6vectorIdSaIdEEaSERKS1_:
.LFB3820:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	%rdi, %rbx
	subq	$16, %rsp
	.cfi_def_cfa_offset 64
	cmpq	%rdi, %rsi
	je	.L485
	movq	%rsi, %r12
	movq	8(%r12), %r15
	movq	(%rsi), %rsi
	movq	%r15, %rbp
	movq	(%rdi), %r14
	movq	16(%rdi), %rcx
	subq	%rsi, %rbp
	movq	%rbp, %rax
	subq	%r14, %rcx
	sarq	$3, %rax
	sarq	$3, %rcx
	cmpq	%rcx, %rax
	ja	.L488
	movq	8(%rdi), %rdi
	movq	%rdi, %rdx
	subq	%r14, %rdx
	movq	%rdx, %rcx
	sarq	$3, %rcx
	cmpq	%rcx, %rax
	ja	.L475
	cmpq	%r15, %rsi
	je	.L487
	movq	%rbp, %rdx
	movq	%r14, %rdi
	call	memmove@PLT
	addq	(%rbx), %rbp
	movq	%rbp, 8(%rbx)
.L485:
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	movq	%rbx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L488:
	.cfi_restore_state
	xorl	%r12d, %r12d
	testq	%rax, %rax
	je	.L470
	movabsq	$2305843009213693951, %rdx
	cmpq	%rdx, %rax
	ja	.L489
	movq	%rbp, %rdi
	movq	%rsi, 8(%rsp)
	call	_Znwm@PLT
	movq	(%rbx), %r14
	movq	8(%rsp), %rsi
	movq	%rax, %r12
.L470:
	cmpq	%r15, %rsi
	je	.L472
	movq	%rbp, %rdx
	movq	%r12, %rdi
	call	memmove@PLT
.L472:
	testq	%r14, %r14
	je	.L473
	movq	%r14, %rdi
	call	_ZdlPv@PLT
.L473:
	addq	%r12, %rbp
	movq	%r12, (%rbx)
	movq	%rbp, 16(%rbx)
	jmp	.L474
	.p2align 4,,10
	.p2align 3
.L475:
	testq	%rdx, %rdx
	je	.L477
	movq	%r14, %rdi
	call	memmove@PLT
	movq	8(%rbx), %rdi
	movq	(%rbx), %r14
	movq	%rdi, %rdx
	movq	8(%r12), %r15
	movq	(%r12), %rsi
	subq	%r14, %rdx
.L477:
	addq	%rdx, %rsi
	cmpq	%r15, %rsi
	jne	.L478
.L487:
	addq	%r14, %rbp
.L474:
	movq	%rbp, 8(%rbx)
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	movq	%rbx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L478:
	.cfi_restore_state
	movq	%r15, %rdx
	subq	%rsi, %rdx
	call	memmove@PLT
	addq	(%rbx), %rbp
	jmp	.L474
.L489:
	call	_ZSt17__throw_bad_allocv@PLT
	.cfi_endproc
.LFE3820:
	.size	_ZNSt6vectorIdSaIdEEaSERKS1_, .-_ZNSt6vectorIdSaIdEEaSERKS1_
	.section	.text._ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_,"axG",@progbits,_ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_
	.type	_ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_, @function
_ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_:
.LFB3446:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA3446
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	addq	$-128, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdi, 64(%rsp)
	movq	%rsi, 72(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 120(%rsp)
	xorl	%eax, %eax
	testq	%rdx, %rdx
	je	.L490
	movq	16(%rdi), %rax
	movq	8(%rdi), %rbx
	movq	%rax, 56(%rsp)
	subq	%rbx, %rax
	movq	%rdx, %r14
	sarq	$3, %rax
	movabsq	$-6148914691236517205, %rdx
	imulq	%rdx, %rax
	movq	%rcx, %r12
	movq	%rsi, %r15
	cmpq	%r14, %rax
	jnb	.L640
	movq	(%rdi), %rax
	movabsq	$768614336404564650, %rsi
	subq	%rax, %rbx
	sarq	$3, %rbx
	imulq	%rdx, %rbx
	movq	%rsi, %rcx
	subq	%rbx, %rcx
	cmpq	%rcx, %r14
	ja	.L641
	cmpq	%rbx, %r14
	movq	%rbx, %rcx
	cmovnb	%r14, %rcx
	addq	%rcx, %rbx
	movq	%rbx, %rdi
	movq	%rbx, 16(%rsp)
	movq	72(%rsp), %rbx
	setc	%cl
	subq	%rax, %rbx
	movq	%rbx, %rax
	sarq	$3, %rax
	imulq	%rdx, %rax
	movzbl	%cl, %ecx
	movq	%rax, (%rsp)
	testq	%rcx, %rcx
	jne	.L568
	cmpq	%rsi, %rdi
	ja	.L568
	movq	$0, 24(%rsp)
	testq	%rdi, %rdi
	jne	.L642
.L530:
	addq	24(%rsp), %rbx
	movq	%rbx, 8(%rsp)
	movq	%r14, 56(%rsp)
	movq	(%r12), %r8
	movq	8(%r12), %r9
	.p2align 4,,10
	.p2align 3
.L537:
	movq	%r9, %r13
	subq	%r8, %r13
	movq	%r13, %rax
	sarq	$3, %rax
	movq	$0, (%rbx)
	movq	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	je	.L643
	movabsq	$2305843009213693951, %rsi
	cmpq	%rsi, %rax
	ja	.L644
	movq	%r13, %rdi
.LEHB7:
	call	_Znwm@PLT
.LEHE7:
	movq	8(%r12), %r9
	movq	(%r12), %r8
	movq	%r9, %r10
	movq	%rax, %rcx
	subq	%r8, %r10
.L532:
	addq	%rcx, %r13
	movq	%rcx, (%rbx)
	movq	%rcx, 8(%rbx)
	movq	%r13, 16(%rbx)
	cmpq	%r9, %r8
	je	.L534
	movq	%r10, %rdx
	movq	%r8, %rsi
	movq	%rcx, %rdi
	movq	%r9, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%r8, 48(%rsp)
	call	memmove@PLT
	movq	40(%rsp), %r10
	movq	%rax, %rcx
	addq	%r10, %rcx
	movq	%rcx, 8(%rbx)
	addq	$24, %rbx
	decq	56(%rsp)
	movq	48(%rsp), %r8
	movq	32(%rsp), %r9
	jne	.L537
.L535:
	movq	64(%rsp), %rax
	movq	72(%rsp), %rdi
	movq	(%rax), %r12
	movq	24(%rsp), %rax
	movq	%r12, %rdx
	cmpq	%r12, 72(%rsp)
	je	.L539
	.p2align 4,,10
	.p2align 3
.L547:
	movq	$0, 8(%rax)
	movq	$0, 16(%rax)
	movq	$0, (%rax)
	addq	$24, %rdx
	addq	$24, %rax
	movq	-24(%rdx), %rcx
	movq	%rcx, -24(%rax)
	movq	$0, -24(%rdx)
	movq	-16(%rdx), %rsi
	movq	-16(%rax), %rcx
	movq	%rsi, -16(%rax)
	movq	%rcx, -16(%rdx)
	movq	-8(%rdx), %rsi
	movq	-8(%rax), %rcx
	movq	%rsi, -8(%rax)
	movq	%rcx, -8(%rdx)
	cmpq	%rdx, %rdi
	jne	.L547
	movq	72(%rsp), %rax
	movabsq	$768614336404564651, %rdx
	subq	$24, %rax
	subq	%r12, %rax
	shrq	$3, %rax
	imulq	%rdx, %rax
	movabsq	$2305843009213693951, %rdx
	movq	24(%rsp), %rsi
	andq	%rdx, %rax
	leaq	3(%rax,%rax,2), %rax
	leaq	(%rsi,%rax,8), %rax
.L539:
	leaq	(%r14,%r14,2), %rdx
	leaq	(%rax,%rdx,8), %rbx
	movq	64(%rsp), %rax
	movq	8(%rax), %r13
	cmpq	%r13, 72(%rsp)
	je	.L554
	movq	72(%rsp), %rdi
	movq	%r13, %rdx
	subq	%rdi, %rdx
	subq	$24, %rdx
	movabsq	$768614336404564651, %rcx
	shrq	$3, %rdx
	imulq	%rcx, %rdx
	movabsq	$2305843009213693951, %rsi
	movabsq	$2305843009213693948, %rcx
	andq	%rdx, %rsi
	movq	%rdi, %rax
	incq	%rsi
	testq	%rcx, %rdx
	je	.L570
	movq	%rsi, %rcx
	shrq	$2, %rcx
	leaq	(%rcx,%rcx,2), %rcx
	salq	$5, %rcx
	movq	%rbx, %rdx
	addq	%rdi, %rcx
	vpxor	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L552:
	vmovdqu	(%rax), %ymm1
	vmovdqu	32(%rax), %ymm2
	vmovdqu	64(%rax), %ymm3
	vmovdqu	%ymm0, (%rax)
	vmovdqu	%ymm0, 32(%rax)
	vmovdqu	%ymm0, 64(%rax)
	addq	$96, %rax
	vmovdqu	%ymm1, (%rdx)
	vmovdqu	%ymm2, 32(%rdx)
	vmovdqu	%ymm3, 64(%rdx)
	addq	$96, %rdx
	cmpq	%rcx, %rax
	jne	.L552
	movq	%rsi, %rdx
	andq	$-4, %rdx
	leaq	(%rdx,%rdx,2), %rax
	movq	72(%rsp), %r15
	salq	$3, %rax
	addq	%rax, %r15
	addq	%rbx, %rax
	cmpq	%rsi, %rdx
	je	.L645
	vzeroupper
.L551:
	movq	(%r15), %rdx
	movq	%rdx, (%rax)
	movq	$0, (%r15)
	movq	8(%r15), %rdx
	movq	%rdx, 8(%rax)
	movq	$0, 8(%r15)
	movq	16(%r15), %rdx
	movq	%rdx, 16(%rax)
	leaq	24(%r15), %rdx
	movq	$0, 16(%r15)
	cmpq	%rdx, %r13
	je	.L555
	movq	24(%r15), %rdx
	movq	%rdx, 24(%rax)
	movq	$0, 24(%r15)
	movq	32(%r15), %rdx
	movq	%rdx, 32(%rax)
	movq	$0, 32(%r15)
	movq	40(%r15), %rdx
	movq	%rdx, 40(%rax)
	leaq	48(%r15), %rdx
	movq	$0, 40(%r15)
	cmpq	%rdx, %r13
	je	.L555
	movq	48(%r15), %rdx
	movq	%rdx, 48(%rax)
	movq	$0, 48(%r15)
	movq	56(%r15), %rdx
	movq	%rdx, 56(%rax)
	movq	$0, 56(%r15)
	movq	64(%r15), %rdx
	movq	%rdx, 64(%rax)
	leaq	72(%r15), %rdx
	movq	$0, 64(%r15)
	cmpq	%rdx, %r13
	je	.L555
	movq	72(%r15), %rdx
	movq	%rdx, 72(%rax)
	movq	$0, 72(%r15)
	movq	80(%r15), %rdx
	movq	%rdx, 80(%rax)
	movq	$0, 80(%r15)
	movq	88(%r15), %rdx
	movq	%rdx, 88(%rax)
	movq	$0, 88(%r15)
.L555:
	leaq	(%rsi,%rsi,2), %rax
	leaq	(%rbx,%rax,8), %rbx
.L554:
	cmpq	%r13, %r12
	je	.L549
	.p2align 4,,10
	.p2align 3
.L550:
	movq	(%r12), %rdi
	testq	%rdi, %rdi
	je	.L556
	addq	$24, %r12
	call	_ZdlPv@PLT
	cmpq	%r12, %r13
	jne	.L550
.L557:
	movq	64(%rsp), %rax
	movq	(%rax), %r13
.L549:
	testq	%r13, %r13
	je	.L559
	movq	%r13, %rdi
	call	_ZdlPv@PLT
.L559:
	movq	64(%rsp), %rdi
	movq	24(%rsp), %rax
	movq	%rbx, 8(%rdi)
	movq	%rax, (%rdi)
	addq	16(%rsp), %rax
	movq	%rax, 16(%rdi)
.L490:
	movq	120(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L646
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L640:
	.cfi_restore_state
	movq	8(%rcx), %rax
	movq	(%rcx), %rsi
	movq	%rax, %r13
	subq	%rsi, %r13
	movq	%r13, %rdx
	sarq	$3, %rdx
	movq	%rdi, 80(%rsp)
	movq	$0, 88(%rsp)
	movq	$0, 96(%rsp)
	movq	$0, 104(%rsp)
	je	.L647
	movabsq	$2305843009213693951, %rax
	cmpq	%rax, %rdx
	ja	.L648
	movq	%r13, %rdi
.LEHB8:
	call	_Znwm@PLT
.LEHE8:
	movq	%rax, %rcx
	movq	64(%rsp), %rax
	movq	(%r12), %rsi
	movq	8(%rax), %rbx
	movq	8(%r12), %rax
	movq	%rax, %r12
	subq	%rsi, %r12
.L494:
	addq	%rcx, %r13
	movq	%rcx, 88(%rsp)
	movq	%rcx, 96(%rsp)
	movq	%r13, 104(%rsp)
	cmpq	%rax, %rsi
	je	.L496
	movq	%rcx, %rdi
	movq	%r12, %rdx
	call	memmove@PLT
	movq	%rax, %rcx
.L496:
	movq	%rbx, %rax
	subq	72(%rsp), %rax
	movq	%rax, 56(%rsp)
	movabsq	$-6148914691236517205, %rdx
	sarq	$3, %rax
	imulq	%rdx, %rax
	addq	%r12, %rcx
	movq	%rcx, 96(%rsp)
	cmpq	%rax, %r14
	jnb	.L497
	leaq	(%r14,%r14,2), %r12
	salq	$3, %r12
	movq	%rbx, %r13
	subq	%r12, %r13
	movq	%r13, %rdx
	movq	%rbx, %rax
	cmpq	%rbx, %r13
	je	.L502
	.p2align 4,,10
	.p2align 3
.L501:
	movq	$0, 8(%rax)
	movq	$0, 16(%rax)
	movq	$0, (%rax)
	addq	$24, %rdx
	addq	$24, %rax
	movq	-24(%rdx), %rcx
	movq	%rcx, -24(%rax)
	movq	$0, -24(%rdx)
	movq	-16(%rdx), %rsi
	movq	-16(%rax), %rcx
	movq	%rsi, -16(%rax)
	movq	%rcx, -16(%rdx)
	movq	-8(%rdx), %rsi
	movq	-8(%rax), %rcx
	movq	%rsi, -8(%rax)
	movq	%rcx, -8(%rdx)
	cmpq	%rdx, %rbx
	jne	.L501
.L502:
	movq	64(%rsp), %rsi
	leaq	(%rbx,%r12), %rax
	movq	%rax, 8(%rsi)
	movq	%r13, %rax
	subq	72(%rsp), %rax
	movq	%rax, %r14
	sarq	$3, %r14
	movabsq	$-6148914691236517205, %rdx
	imulq	%rdx, %r14
	testq	%rax, %rax
	jle	.L499
	.p2align 4,,10
	.p2align 3
.L500:
	subq	$24, %rbx
	movq	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	movq	(%rbx), %rdi
	movq	$0, (%rbx)
	subq	$24, %r13
	movq	0(%r13), %rax
	movq	%rax, (%rbx)
	movq	$0, 0(%r13)
	movq	8(%r13), %rdx
	movq	8(%rbx), %rax
	movq	%rdx, 8(%rbx)
	movq	%rax, 8(%r13)
	movq	16(%r13), %rdx
	movq	16(%rbx), %rax
	movq	%rdx, 16(%rbx)
	movq	%rax, 16(%r13)
	testq	%rdi, %rdi
	je	.L505
	call	_ZdlPv@PLT
.L505:
	decq	%r14
	jne	.L500
.L499:
	movq	72(%rsp), %rax
	addq	%rax, %r12
	cmpq	%r12, %rax
	je	.L506
	leaq	88(%rsp), %rbx
	.p2align 4,,10
	.p2align 3
.L504:
	movq	%rbx, %rsi
	movq	%r15, %rdi
.LEHB9:
	call	_ZNSt6vectorIdSaIdEEaSERKS1_
.LEHE9:
	addq	$24, %r15
	cmpq	%r15, %r12
	jne	.L504
.L506:
	movq	88(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L490
	call	_ZdlPv@PLT
	jmp	.L490
	.p2align 4,,10
	.p2align 3
.L568:
	movq	$-16, 16(%rsp)
.L529:
	movq	16(%rsp), %rdi
.LEHB10:
	call	_Znwm@PLT
.LEHE10:
	movq	%rax, 24(%rsp)
	jmp	.L530
	.p2align 4,,10
	.p2align 3
.L647:
	movq	%r13, %r12
	xorl	%ecx, %ecx
	jmp	.L494
	.p2align 4,,10
	.p2align 3
.L556:
	addq	$24, %r12
	cmpq	%r12, %r13
	jne	.L550
	jmp	.L557
	.p2align 4,,10
	.p2align 3
.L534:
	addq	%r10, %rcx
	movq	%rcx, 8(%rbx)
	addq	$24, %rbx
	decq	56(%rsp)
	jne	.L537
	jmp	.L535
	.p2align 4,,10
	.p2align 3
.L643:
	movq	%r13, %r10
	xorl	%ecx, %ecx
	jmp	.L532
	.p2align 4,,10
	.p2align 3
.L497:
	movq	%rbx, %r12
	subq	%rax, %r14
	jne	.L513
	jmp	.L507
	.p2align 4,,10
	.p2align 3
.L508:
	movabsq	$2305843009213693951, %rsi
	cmpq	%rsi, %rax
	ja	.L649
	movq	%r13, %rdi
.LEHB11:
	call	_Znwm@PLT
.LEHE11:
	movq	%rax, %rcx
.L509:
	addq	%rcx, %r13
	movq	%r13, 16(%r12)
	movq	%rcx, (%r12)
	movq	%rcx, 8(%r12)
	movq	96(%rsp), %rax
	movq	88(%rsp), %rsi
	movq	%rax, %r13
	subq	%rsi, %r13
	cmpq	%rsi, %rax
	je	.L638
	movq	%rcx, %rdi
	movq	%r13, %rdx
	call	memmove@PLT
	movq	%rax, %rcx
.L638:
	addq	%r13, %rcx
	movq	%rcx, 8(%r12)
	addq	$24, %r12
	decq	%r14
	je	.L507
.L513:
	movq	96(%rsp), %r13
	subq	88(%rsp), %r13
	movq	%r13, %rax
	sarq	$3, %rax
	movq	$0, (%r12)
	movq	$0, 8(%r12)
	movq	$0, 16(%r12)
	jne	.L508
	xorl	%ecx, %ecx
	jmp	.L509
	.p2align 4,,10
	.p2align 3
.L507:
	movq	64(%rsp), %rax
	movq	56(%rsp), %rsi
	movq	%r12, 8(%rax)
	movq	72(%rsp), %rax
	addq	%r12, %rsi
	cmpq	%rbx, %rax
	je	.L650
	.p2align 4,,10
	.p2align 3
.L522:
	movq	$0, 8(%r12)
	movq	$0, 16(%r12)
	movq	$0, (%r12)
	addq	$24, %rax
	addq	$24, %r12
	movq	-24(%rax), %rdx
	movq	%rdx, -24(%r12)
	movq	$0, -24(%rax)
	movq	-16(%rax), %rcx
	movq	-16(%r12), %rdx
	movq	%rcx, -16(%r12)
	movq	%rdx, -16(%rax)
	movq	-8(%rax), %rcx
	movq	-8(%r12), %rdx
	movq	%rcx, -8(%r12)
	movq	%rdx, -8(%rax)
	cmpq	%rax, %rbx
	jne	.L522
	movq	64(%rsp), %rax
	leaq	88(%rsp), %r12
	movq	%rsi, 8(%rax)
	.p2align 4,,10
	.p2align 3
.L524:
	movq	%r12, %rsi
	movq	%r15, %rdi
.LEHB12:
	call	_ZNSt6vectorIdSaIdEEaSERKS1_
.LEHE12:
	addq	$24, %r15
	cmpq	%rbx, %r15
	jne	.L524
	jmp	.L506
	.p2align 4,,10
	.p2align 3
.L645:
	vzeroupper
	jmp	.L555
	.p2align 4,,10
	.p2align 3
.L642:
	leaq	(%rdi,%rdi,2), %rax
	salq	$3, %rax
	movq	%rax, 16(%rsp)
	jmp	.L529
.L570:
	movq	%rbx, %rax
	jmp	.L551
.L650:
	movq	64(%rsp), %rax
	movq	%rsi, 8(%rax)
	jmp	.L506
.L644:
.LEHB13:
	call	_ZSt17__throw_bad_allocv@PLT
.LEHE13:
.L646:
	call	__stack_chk_fail@PLT
.L649:
.LEHB14:
	call	_ZSt17__throw_bad_allocv@PLT
.LEHE14:
.L648:
.LEHB15:
	call	_ZSt17__throw_bad_allocv@PLT
.L641:
	leaq	.LC8(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE15:
.L574:
	jmp	.L516
.L571:
	movq	%rax, %rbx
	jmp	.L521
.L576:
	jmp	.L540
.L516:
	movq	%rax, %rdi
	vzeroupper
	call	__cxa_begin_catch@PLT
.L519:
	cmpq	%r12, %rbx
	jne	.L651
.LEHB16:
	call	__cxa_rethrow@PLT
.LEHE16:
.L520:
	vzeroupper
	call	__cxa_end_catch@PLT
.L521:
	movq	88(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L634
	vzeroupper
	call	_ZdlPv@PLT
.L639:
	movq	%rbx, %rdi
.LEHB17:
	call	_Unwind_Resume@PLT
.LEHE17:
.L540:
	movq	%rax, %rdi
	vzeroupper
	call	__cxa_begin_catch@PLT
	movq	8(%rsp), %r12
.L543:
	cmpq	%rbx, %r12
	jne	.L652
.LEHB18:
	call	__cxa_rethrow@PLT
.LEHE18:
.L634:
	vzeroupper
	jmp	.L639
.L651:
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.L518
	call	_ZdlPv@PLT
.L518:
	addq	$24, %rbx
	jmp	.L519
.L652:
	movq	(%r12), %rdi
	testq	%rdi, %rdi
	je	.L542
	call	_ZdlPv@PLT
.L542:
	addq	$24, %r12
	jmp	.L543
.L573:
	movq	%rax, %rbx
	jmp	.L520
.L575:
	movq	%rax, %rbx
.L544:
	vzeroupper
	call	__cxa_end_catch@PLT
	movq	%rbx, %rdi
	call	__cxa_begin_catch@PLT
	cmpq	$0, 24(%rsp)
	je	.L653
	movq	24(%rsp), %rdi
	call	_ZdlPv@PLT
.L561:
.LEHB19:
	call	__cxa_rethrow@PLT
.LEHE19:
.L653:
	addq	(%rsp), %r14
	imulq	$24, %r14, %r14
	movq	8(%rsp), %rbx
.L563:
	cmpq	%rbx, %r14
	je	.L561
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.L562
	call	_ZdlPv@PLT
.L562:
	addq	$24, %rbx
	jmp	.L563
.L572:
	movq	%rax, %rbx
.L564:
	vzeroupper
	call	__cxa_end_catch@PLT
	jmp	.L639
	.cfi_endproc
.LFE3446:
	.section	.gcc_except_table
	.align 4
.LLSDA3446:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT3446-.LLSDATTD3446
.LLSDATTD3446:
	.byte	0x1
	.uleb128 .LLSDACSE3446-.LLSDACSB3446
.LLSDACSB3446:
	.uleb128 .LEHB7-.LFB3446
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L576-.LFB3446
	.uleb128 0x1
	.uleb128 .LEHB8-.LFB3446
	.uleb128 .LEHE8-.LEHB8
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB9-.LFB3446
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L571-.LFB3446
	.uleb128 0
	.uleb128 .LEHB10-.LFB3446
	.uleb128 .LEHE10-.LEHB10
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB11-.LFB3446
	.uleb128 .LEHE11-.LEHB11
	.uleb128 .L574-.LFB3446
	.uleb128 0x1
	.uleb128 .LEHB12-.LFB3446
	.uleb128 .LEHE12-.LEHB12
	.uleb128 .L571-.LFB3446
	.uleb128 0
	.uleb128 .LEHB13-.LFB3446
	.uleb128 .LEHE13-.LEHB13
	.uleb128 .L576-.LFB3446
	.uleb128 0x1
	.uleb128 .LEHB14-.LFB3446
	.uleb128 .LEHE14-.LEHB14
	.uleb128 .L574-.LFB3446
	.uleb128 0x1
	.uleb128 .LEHB15-.LFB3446
	.uleb128 .LEHE15-.LEHB15
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB16-.LFB3446
	.uleb128 .LEHE16-.LEHB16
	.uleb128 .L573-.LFB3446
	.uleb128 0
	.uleb128 .LEHB17-.LFB3446
	.uleb128 .LEHE17-.LEHB17
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB18-.LFB3446
	.uleb128 .LEHE18-.LEHB18
	.uleb128 .L575-.LFB3446
	.uleb128 0x3
	.uleb128 .LEHB19-.LFB3446
	.uleb128 .LEHE19-.LEHB19
	.uleb128 .L572-.LFB3446
	.uleb128 0
.LLSDACSE3446:
	.byte	0x1
	.byte	0
	.byte	0
	.byte	0x7d
	.align 4
	.long	0

.LLSDATT3446:
	.section	.text._ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_,"axG",@progbits,_ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_,comdat
	.size	_ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_, .-_ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_
	.section	.text.unlikely
	.align 2
.LCOLDB9:
	.text
.LHOTB9:
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.type	_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, @function
_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE:
.LFB2885:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2885
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, %rbx
	subq	$568, %rsp
	.cfi_def_cfa_offset 624
	leaq	32(%rsp), %r13
	leaq	256(%r13), %rdi
	movq	%fs:40, %rax
	movq	%rax, 552(%rsp)
	xorl	%eax, %eax
	call	_ZNSt8ios_baseC2Ev@PLT
	leaq	16+_ZTVSt9basic_iosIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 288(%rsp)
	xorl	%eax, %eax
	movw	%ax, 512(%rsp)
	movq	8+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %r15
	movq	16+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rcx
	movq	-24(%r15), %rax
	movq	$0, 504(%rsp)
	movq	$0, 520(%rsp)
	movq	$0, 528(%rsp)
	movq	$0, 536(%rsp)
	movq	$0, 544(%rsp)
	movq	%r15, 32(%rsp)
	movq	%rcx, 32(%rsp,%rax)
	movq	$0, 40(%rsp)
	xorl	%esi, %esi
	movq	-24(%r15), %rdi
	addq	%r13, %rdi
.LEHB20:
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@PLT
.LEHE20:
	leaq	24+_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 32(%rsp)
	leaq	16(%r13), %rdi
	addq	$40, %rax
	movq	%rax, 288(%rsp)
.LEHB21:
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@PLT
.LEHE21:
	leaq	16(%r13), %rsi
	leaq	256(%r13), %rdi
.LEHB22:
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@PLT
	movq	(%rbx), %rsi
	leaq	16(%r13), %rdi
	movl	$8, %edx
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@PLT
	movq	32(%rsp), %rdx
	movq	-24(%rdx), %rdi
	addq	%r13, %rdi
	testq	%rax, %rax
	je	.L701
	xorl	%esi, %esi
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
.LEHE22:
.L656:
	movq	%rbp, %rsi
	movq	%r13, %rdi
.LEHB23:
	call	_ZNSirsERi@PLT
	leaq	4(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSirsERi@PLT
	leaq	8(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSirsERi@PLT
	movslq	4(%rbp), %rbx
	movq	$0, (%rsp)
	movq	$0, 8(%rsp)
	movq	$0, 16(%rsp)
	testq	%rbx, %rbx
	je	.L681
	movabsq	$2305843009213693951, %rax
	cmpq	%rax, %rbx
	ja	.L702
	salq	$3, %rbx
	movq	%rbx, %rdi
	call	_Znwm@PLT
	leaq	(%rax,%rbx), %r12
	movq	%rbx, %rdx
	xorl	%esi, %esi
	movq	%rax, %rdi
	movq	%rax, (%rsp)
	movq	%r12, 16(%rsp)
	call	memset@PLT
	movq	%rax, %rcx
.L661:
	movq	24(%rbp), %r14
	movq	16(%rbp), %rsi
	movq	%r14, %rax
	subq	%rsi, %rax
	sarq	$3, %rax
	movabsq	$-6148914691236517205, %rdi
	imulq	%rdi, %rax
	movslq	0(%rbp), %rdx
	movq	%r12, 8(%rsp)
	cmpq	%rax, %rdx
	ja	.L703
	jb	.L704
.L664:
	testq	%rcx, %rcx
	je	.L669
	movq	%rcx, %rdi
	call	_ZdlPv@PLT
.L669:
	movl	0(%rbp), %edx
	testl	%edx, %edx
	jle	.L674
	movl	4(%rbp), %eax
	xorl	%r14d, %r14d
	.p2align 4,,10
	.p2align 3
.L675:
	testl	%eax, %eax
	jle	.L672
	leaq	(%r14,%r14,2), %r12
	salq	$3, %r12
	xorl	%ebx, %ebx
	.p2align 4,,10
	.p2align 3
.L673:
	movq	16(%rbp), %rax
	movq	%r13, %rdi
	movq	(%rax,%r12), %rax
	leaq	(%rax,%rbx,8), %rsi
	call	_ZNSi10_M_extractIdEERSiRT_@PLT
	movl	4(%rbp), %eax
	incq	%rbx
	cmpl	%ebx, %eax
	jg	.L673
	movl	0(%rbp), %edx
.L672:
	incq	%r14
	cmpl	%r14d, %edx
	jg	.L675
.L674:
	leaq	16(%r13), %rdi
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@PLT
.LEHE23:
	testq	%rax, %rax
	je	.L705
.L676:
	leaq	24+_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 32(%rsp)
	addq	$40, %rax
	movq	%rax, 288(%rsp)
	leaq	16(%r13), %rdi
	leaq	16+_ZTVSt13basic_filebufIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 48(%rsp)
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@PLT
	leaq	120(%r13), %rdi
	call	_ZNSt12__basic_fileIcED1Ev@PLT
	leaq	16+_ZTVSt15basic_streambufIcSt11char_traitsIcEE(%rip), %rax
	leaq	72(%r13), %rdi
	movq	%rax, 48(%rsp)
	call	_ZNSt6localeD1Ev@PLT
	movq	-24(%r15), %rax
	movq	16+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rcx
	movq	%r15, 32(%rsp)
	leaq	256(%r13), %rdi
	movq	%rcx, 32(%rsp,%rax)
	leaq	16+_ZTVSt9basic_iosIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 288(%rsp)
	movq	$0, 40(%rsp)
	call	_ZNSt8ios_baseD2Ev@PLT
	movq	552(%rsp), %rcx
	xorq	%fs:40, %rcx
	movl	$1, %eax
	jne	.L706
	addq	$568, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L704:
	.cfi_restore_state
	leaq	(%rdx,%rdx,2), %rax
	leaq	(%rsi,%rax,8), %rbx
	cmpq	%rbx, %r14
	je	.L664
	movq	%rbx, %r12
	.p2align 4,,10
	.p2align 3
.L668:
	movq	(%r12), %rdi
	testq	%rdi, %rdi
	je	.L665
	addq	$24, %r12
	call	_ZdlPv@PLT
	cmpq	%r14, %r12
	jne	.L668
.L666:
	movq	%rbx, 24(%rbp)
	jmp	.L700
.L703:
	movq	%rsp, %rcx
	subq	%rax, %rdx
	leaq	16(%rbp), %rdi
	movq	%r14, %rsi
.LEHB24:
	call	_ZNSt6vectorIS_IdSaIdEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_
.LEHE24:
.L700:
	movq	(%rsp), %rcx
	jmp	.L664
.L701:
	movl	32(%rdi), %esi
	orl	$4, %esi
.LEHB25:
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
.LEHE25:
	jmp	.L656
.L681:
	xorl	%ecx, %ecx
	xorl	%r12d, %r12d
	jmp	.L661
.L665:
	addq	$24, %r12
	cmpq	%r12, %r14
	jne	.L668
	jmp	.L666
.L705:
	movq	32(%rsp), %rax
	movq	-24(%rax), %rdi
	addq	%r13, %rdi
	movl	32(%rdi), %esi
	orl	$4, %esi
.LEHB26:
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
	jmp	.L676
.L706:
	call	__stack_chk_fail@PLT
.L702:
	call	_ZSt17__throw_bad_allocv@PLT
.LEHE26:
.L683:
	movq	%rax, %rbx
	jmp	.L677
.L686:
	movq	%rax, %rbx
	jmp	.L658
.L685:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L659
.L682:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L679
.L684:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L660
	.section	.gcc_except_table
.LLSDA2885:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2885-.LLSDACSB2885
.LLSDACSB2885:
	.uleb128 .LEHB20-.LFB2885
	.uleb128 .LEHE20-.LEHB20
	.uleb128 .L684-.LFB2885
	.uleb128 0
	.uleb128 .LEHB21-.LFB2885
	.uleb128 .LEHE21-.LEHB21
	.uleb128 .L685-.LFB2885
	.uleb128 0
	.uleb128 .LEHB22-.LFB2885
	.uleb128 .LEHE22-.LEHB22
	.uleb128 .L686-.LFB2885
	.uleb128 0
	.uleb128 .LEHB23-.LFB2885
	.uleb128 .LEHE23-.LEHB23
	.uleb128 .L682-.LFB2885
	.uleb128 0
	.uleb128 .LEHB24-.LFB2885
	.uleb128 .LEHE24-.LEHB24
	.uleb128 .L683-.LFB2885
	.uleb128 0
	.uleb128 .LEHB25-.LFB2885
	.uleb128 .LEHE25-.LEHB25
	.uleb128 .L686-.LFB2885
	.uleb128 0
	.uleb128 .LEHB26-.LFB2885
	.uleb128 .LEHE26-.LEHB26
	.uleb128 .L682-.LFB2885
	.uleb128 0
.LLSDACSE2885:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC2885
	.type	_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE.cold.185, @function
_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE.cold.185:
.LFSB2885:
.L677:
	.cfi_def_cfa_offset 624
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	.cfi_offset 12, -40
	.cfi_offset 13, -32
	.cfi_offset 14, -24
	.cfi_offset 15, -16
	movq	(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L697
	vzeroupper
	call	_ZdlPv@PLT
.L679:
	movq	%r13, %rdi
	call	_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@PLT
	movq	%rbx, %rdi
.LEHB27:
	call	_Unwind_Resume@PLT
.L658:
	leaq	16(%r13), %rdi
	vzeroupper
	call	_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@PLT
.L659:
	movq	-24(%r15), %rdx
	movq	16+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%r15, 32(%rsp)
	movq	%rax, 32(%rsp,%rdx)
	movq	$0, 40(%rsp)
.L660:
	leaq	16+_ZTVSt9basic_iosIcSt11char_traitsIcEE(%rip), %rax
	leaq	256(%r13), %rdi
	movq	%rax, 288(%rsp)
	call	_ZNSt8ios_baseD2Ev@PLT
	movq	%rbx, %rdi
	call	_Unwind_Resume@PLT
.LEHE27:
.L697:
	vzeroupper
	jmp	.L679
	.cfi_endproc
.LFE2885:
	.section	.gcc_except_table
.LLSDAC2885:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC2885-.LLSDACSBC2885
.LLSDACSBC2885:
	.uleb128 .LEHB27-.LCOLDB9
	.uleb128 .LEHE27-.LEHB27
	.uleb128 0
	.uleb128 0
.LLSDACSEC2885:
	.section	.text.unlikely
	.text
	.size	_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, .-_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.section	.text.unlikely
	.size	_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE.cold.185, .-_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE.cold.185
.LCOLDE9:
	.text
.LHOTE9:
	.section	.text._ZNSt6vectorIiSaIiEEaSERKS1_,"axG",@progbits,_ZNSt6vectorIiSaIiEEaSERKS1_,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZNSt6vectorIiSaIiEEaSERKS1_
	.type	_ZNSt6vectorIiSaIiEEaSERKS1_, @function
_ZNSt6vectorIiSaIiEEaSERKS1_:
.LFB3842:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	%rdi, %rbx
	subq	$16, %rsp
	.cfi_def_cfa_offset 64
	cmpq	%rdi, %rsi
	je	.L726
	movq	%rsi, %r12
	movq	8(%r12), %r15
	movq	(%rsi), %rsi
	movq	%r15, %rbp
	movq	(%rdi), %r14
	movq	16(%rdi), %rcx
	subq	%rsi, %rbp
	movq	%rbp, %rax
	subq	%r14, %rcx
	sarq	$2, %rax
	sarq	$2, %rcx
	cmpq	%rcx, %rax
	ja	.L729
	movq	8(%rdi), %rdi
	movq	%rdi, %rdx
	subq	%r14, %rdx
	movq	%rdx, %rcx
	sarq	$2, %rcx
	cmpq	%rcx, %rax
	ja	.L716
	cmpq	%r15, %rsi
	je	.L728
	movq	%rbp, %rdx
	movq	%r14, %rdi
	call	memmove@PLT
	addq	(%rbx), %rbp
	movq	%rbp, 8(%rbx)
.L726:
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	movq	%rbx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L729:
	.cfi_restore_state
	xorl	%r12d, %r12d
	testq	%rax, %rax
	je	.L711
	movabsq	$4611686018427387903, %rdx
	cmpq	%rdx, %rax
	ja	.L730
	movq	%rbp, %rdi
	movq	%rsi, 8(%rsp)
	call	_Znwm@PLT
	movq	(%rbx), %r14
	movq	8(%rsp), %rsi
	movq	%rax, %r12
.L711:
	cmpq	%r15, %rsi
	je	.L713
	movq	%rbp, %rdx
	movq	%r12, %rdi
	call	memmove@PLT
.L713:
	testq	%r14, %r14
	je	.L714
	movq	%r14, %rdi
	call	_ZdlPv@PLT
.L714:
	addq	%r12, %rbp
	movq	%r12, (%rbx)
	movq	%rbp, 16(%rbx)
	jmp	.L715
	.p2align 4,,10
	.p2align 3
.L716:
	testq	%rdx, %rdx
	je	.L718
	movq	%r14, %rdi
	call	memmove@PLT
	movq	8(%rbx), %rdi
	movq	(%rbx), %r14
	movq	%rdi, %rdx
	movq	8(%r12), %r15
	movq	(%r12), %rsi
	subq	%r14, %rdx
.L718:
	addq	%rdx, %rsi
	cmpq	%r15, %rsi
	jne	.L719
.L728:
	addq	%r14, %rbp
.L715:
	movq	%rbp, 8(%rbx)
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	movq	%rbx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L719:
	.cfi_restore_state
	movq	%r15, %rdx
	subq	%rsi, %rdx
	call	memmove@PLT
	addq	(%rbx), %rbp
	jmp	.L715
.L730:
	call	_ZSt17__throw_bad_allocv@PLT
	.cfi_endproc
.LFE3842:
	.size	_ZNSt6vectorIiSaIiEEaSERKS1_, .-_ZNSt6vectorIiSaIiEEaSERKS1_
	.section	.text._ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_,"axG",@progbits,_ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_
	.type	_ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_, @function
_ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_:
.LFB3468:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA3468
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	addq	$-128, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdi, 64(%rsp)
	movq	%rsi, 72(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 120(%rsp)
	xorl	%eax, %eax
	testq	%rdx, %rdx
	je	.L731
	movq	16(%rdi), %rax
	movq	8(%rdi), %rbx
	movq	%rax, 56(%rsp)
	subq	%rbx, %rax
	movq	%rdx, %r14
	sarq	$3, %rax
	movabsq	$-6148914691236517205, %rdx
	imulq	%rdx, %rax
	movq	%rcx, %r12
	movq	%rsi, %r15
	cmpq	%r14, %rax
	jnb	.L881
	movq	(%rdi), %rax
	movabsq	$768614336404564650, %rsi
	subq	%rax, %rbx
	sarq	$3, %rbx
	imulq	%rdx, %rbx
	movq	%rsi, %rcx
	subq	%rbx, %rcx
	cmpq	%rcx, %r14
	ja	.L882
	cmpq	%rbx, %r14
	movq	%rbx, %rcx
	cmovnb	%r14, %rcx
	addq	%rcx, %rbx
	movq	%rbx, %rdi
	movq	%rbx, 16(%rsp)
	movq	72(%rsp), %rbx
	setc	%cl
	subq	%rax, %rbx
	movq	%rbx, %rax
	sarq	$3, %rax
	imulq	%rdx, %rax
	movzbl	%cl, %ecx
	movq	%rax, (%rsp)
	testq	%rcx, %rcx
	jne	.L809
	cmpq	%rsi, %rdi
	ja	.L809
	movq	$0, 24(%rsp)
	testq	%rdi, %rdi
	jne	.L883
.L771:
	addq	24(%rsp), %rbx
	movq	%rbx, 8(%rsp)
	movq	%r14, 56(%rsp)
	movq	(%r12), %r8
	movq	8(%r12), %r9
	.p2align 4,,10
	.p2align 3
.L778:
	movq	%r9, %r13
	subq	%r8, %r13
	movq	%r13, %rax
	sarq	$2, %rax
	movq	$0, (%rbx)
	movq	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	je	.L884
	movabsq	$4611686018427387903, %rsi
	cmpq	%rsi, %rax
	ja	.L885
	movq	%r13, %rdi
.LEHB28:
	call	_Znwm@PLT
.LEHE28:
	movq	8(%r12), %r9
	movq	(%r12), %r8
	movq	%r9, %r10
	movq	%rax, %rcx
	subq	%r8, %r10
.L773:
	addq	%rcx, %r13
	movq	%rcx, (%rbx)
	movq	%rcx, 8(%rbx)
	movq	%r13, 16(%rbx)
	cmpq	%r9, %r8
	je	.L775
	movq	%r10, %rdx
	movq	%r8, %rsi
	movq	%rcx, %rdi
	movq	%r9, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%r8, 48(%rsp)
	call	memmove@PLT
	movq	40(%rsp), %r10
	movq	%rax, %rcx
	addq	%r10, %rcx
	movq	%rcx, 8(%rbx)
	addq	$24, %rbx
	decq	56(%rsp)
	movq	48(%rsp), %r8
	movq	32(%rsp), %r9
	jne	.L778
.L776:
	movq	64(%rsp), %rax
	movq	72(%rsp), %rdi
	movq	(%rax), %r12
	movq	24(%rsp), %rax
	movq	%r12, %rdx
	cmpq	%r12, 72(%rsp)
	je	.L780
	.p2align 4,,10
	.p2align 3
.L788:
	movq	$0, 8(%rax)
	movq	$0, 16(%rax)
	movq	$0, (%rax)
	addq	$24, %rdx
	addq	$24, %rax
	movq	-24(%rdx), %rcx
	movq	%rcx, -24(%rax)
	movq	$0, -24(%rdx)
	movq	-16(%rdx), %rsi
	movq	-16(%rax), %rcx
	movq	%rsi, -16(%rax)
	movq	%rcx, -16(%rdx)
	movq	-8(%rdx), %rsi
	movq	-8(%rax), %rcx
	movq	%rsi, -8(%rax)
	movq	%rcx, -8(%rdx)
	cmpq	%rdx, %rdi
	jne	.L788
	movq	72(%rsp), %rax
	movabsq	$768614336404564651, %rdx
	subq	$24, %rax
	subq	%r12, %rax
	shrq	$3, %rax
	imulq	%rdx, %rax
	movabsq	$2305843009213693951, %rdx
	movq	24(%rsp), %rsi
	andq	%rdx, %rax
	leaq	3(%rax,%rax,2), %rax
	leaq	(%rsi,%rax,8), %rax
.L780:
	leaq	(%r14,%r14,2), %rdx
	leaq	(%rax,%rdx,8), %rbx
	movq	64(%rsp), %rax
	movq	8(%rax), %r13
	cmpq	%r13, 72(%rsp)
	je	.L795
	movq	72(%rsp), %rdi
	movq	%r13, %rdx
	subq	%rdi, %rdx
	subq	$24, %rdx
	movabsq	$768614336404564651, %rcx
	shrq	$3, %rdx
	imulq	%rcx, %rdx
	movabsq	$2305843009213693951, %rsi
	movabsq	$2305843009213693948, %rcx
	andq	%rdx, %rsi
	movq	%rdi, %rax
	incq	%rsi
	testq	%rcx, %rdx
	je	.L811
	movq	%rsi, %rcx
	shrq	$2, %rcx
	leaq	(%rcx,%rcx,2), %rcx
	salq	$5, %rcx
	movq	%rbx, %rdx
	addq	%rdi, %rcx
	vpxor	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L793:
	vmovdqu	(%rax), %ymm1
	vmovdqu	32(%rax), %ymm2
	vmovdqu	64(%rax), %ymm3
	vmovdqu	%ymm0, (%rax)
	vmovdqu	%ymm0, 32(%rax)
	vmovdqu	%ymm0, 64(%rax)
	addq	$96, %rax
	vmovdqu	%ymm1, (%rdx)
	vmovdqu	%ymm2, 32(%rdx)
	vmovdqu	%ymm3, 64(%rdx)
	addq	$96, %rdx
	cmpq	%rcx, %rax
	jne	.L793
	movq	%rsi, %rdx
	andq	$-4, %rdx
	leaq	(%rdx,%rdx,2), %rax
	movq	72(%rsp), %r15
	salq	$3, %rax
	addq	%rax, %r15
	addq	%rbx, %rax
	cmpq	%rsi, %rdx
	je	.L886
	vzeroupper
.L792:
	movq	(%r15), %rdx
	movq	%rdx, (%rax)
	movq	$0, (%r15)
	movq	8(%r15), %rdx
	movq	%rdx, 8(%rax)
	movq	$0, 8(%r15)
	movq	16(%r15), %rdx
	movq	%rdx, 16(%rax)
	leaq	24(%r15), %rdx
	movq	$0, 16(%r15)
	cmpq	%rdx, %r13
	je	.L796
	movq	24(%r15), %rdx
	movq	%rdx, 24(%rax)
	movq	$0, 24(%r15)
	movq	32(%r15), %rdx
	movq	%rdx, 32(%rax)
	movq	$0, 32(%r15)
	movq	40(%r15), %rdx
	movq	%rdx, 40(%rax)
	leaq	48(%r15), %rdx
	movq	$0, 40(%r15)
	cmpq	%rdx, %r13
	je	.L796
	movq	48(%r15), %rdx
	movq	%rdx, 48(%rax)
	movq	$0, 48(%r15)
	movq	56(%r15), %rdx
	movq	%rdx, 56(%rax)
	movq	$0, 56(%r15)
	movq	64(%r15), %rdx
	movq	%rdx, 64(%rax)
	leaq	72(%r15), %rdx
	movq	$0, 64(%r15)
	cmpq	%rdx, %r13
	je	.L796
	movq	72(%r15), %rdx
	movq	%rdx, 72(%rax)
	movq	$0, 72(%r15)
	movq	80(%r15), %rdx
	movq	%rdx, 80(%rax)
	movq	$0, 80(%r15)
	movq	88(%r15), %rdx
	movq	%rdx, 88(%rax)
	movq	$0, 88(%r15)
.L796:
	leaq	(%rsi,%rsi,2), %rax
	leaq	(%rbx,%rax,8), %rbx
.L795:
	cmpq	%r13, %r12
	je	.L790
	.p2align 4,,10
	.p2align 3
.L791:
	movq	(%r12), %rdi
	testq	%rdi, %rdi
	je	.L797
	addq	$24, %r12
	call	_ZdlPv@PLT
	cmpq	%r12, %r13
	jne	.L791
.L798:
	movq	64(%rsp), %rax
	movq	(%rax), %r13
.L790:
	testq	%r13, %r13
	je	.L800
	movq	%r13, %rdi
	call	_ZdlPv@PLT
.L800:
	movq	64(%rsp), %rdi
	movq	24(%rsp), %rax
	movq	%rbx, 8(%rdi)
	movq	%rax, (%rdi)
	addq	16(%rsp), %rax
	movq	%rax, 16(%rdi)
.L731:
	movq	120(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L887
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L881:
	.cfi_restore_state
	movq	8(%rcx), %rax
	movq	(%rcx), %rsi
	movq	%rax, %r13
	subq	%rsi, %r13
	movq	%r13, %rdx
	sarq	$2, %rdx
	movq	%rdi, 80(%rsp)
	movq	$0, 88(%rsp)
	movq	$0, 96(%rsp)
	movq	$0, 104(%rsp)
	je	.L888
	movabsq	$4611686018427387903, %rax
	cmpq	%rax, %rdx
	ja	.L889
	movq	%r13, %rdi
.LEHB29:
	call	_Znwm@PLT
.LEHE29:
	movq	%rax, %rcx
	movq	64(%rsp), %rax
	movq	(%r12), %rsi
	movq	8(%rax), %rbx
	movq	8(%r12), %rax
	movq	%rax, %r12
	subq	%rsi, %r12
.L735:
	addq	%rcx, %r13
	movq	%rcx, 88(%rsp)
	movq	%rcx, 96(%rsp)
	movq	%r13, 104(%rsp)
	cmpq	%rax, %rsi
	je	.L737
	movq	%rcx, %rdi
	movq	%r12, %rdx
	call	memmove@PLT
	movq	%rax, %rcx
.L737:
	movq	%rbx, %rax
	subq	72(%rsp), %rax
	movq	%rax, 56(%rsp)
	movabsq	$-6148914691236517205, %rdx
	sarq	$3, %rax
	imulq	%rdx, %rax
	addq	%r12, %rcx
	movq	%rcx, 96(%rsp)
	cmpq	%rax, %r14
	jnb	.L738
	leaq	(%r14,%r14,2), %r12
	salq	$3, %r12
	movq	%rbx, %r13
	subq	%r12, %r13
	movq	%r13, %rdx
	movq	%rbx, %rax
	cmpq	%rbx, %r13
	je	.L743
	.p2align 4,,10
	.p2align 3
.L742:
	movq	$0, 8(%rax)
	movq	$0, 16(%rax)
	movq	$0, (%rax)
	addq	$24, %rdx
	addq	$24, %rax
	movq	-24(%rdx), %rcx
	movq	%rcx, -24(%rax)
	movq	$0, -24(%rdx)
	movq	-16(%rdx), %rsi
	movq	-16(%rax), %rcx
	movq	%rsi, -16(%rax)
	movq	%rcx, -16(%rdx)
	movq	-8(%rdx), %rsi
	movq	-8(%rax), %rcx
	movq	%rsi, -8(%rax)
	movq	%rcx, -8(%rdx)
	cmpq	%rdx, %rbx
	jne	.L742
.L743:
	movq	64(%rsp), %rsi
	leaq	(%rbx,%r12), %rax
	movq	%rax, 8(%rsi)
	movq	%r13, %rax
	subq	72(%rsp), %rax
	movq	%rax, %r14
	sarq	$3, %r14
	movabsq	$-6148914691236517205, %rdx
	imulq	%rdx, %r14
	testq	%rax, %rax
	jle	.L740
	.p2align 4,,10
	.p2align 3
.L741:
	subq	$24, %rbx
	movq	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	movq	(%rbx), %rdi
	movq	$0, (%rbx)
	subq	$24, %r13
	movq	0(%r13), %rax
	movq	%rax, (%rbx)
	movq	$0, 0(%r13)
	movq	8(%r13), %rdx
	movq	8(%rbx), %rax
	movq	%rdx, 8(%rbx)
	movq	%rax, 8(%r13)
	movq	16(%r13), %rdx
	movq	16(%rbx), %rax
	movq	%rdx, 16(%rbx)
	movq	%rax, 16(%r13)
	testq	%rdi, %rdi
	je	.L746
	call	_ZdlPv@PLT
.L746:
	decq	%r14
	jne	.L741
.L740:
	movq	72(%rsp), %rax
	addq	%rax, %r12
	cmpq	%r12, %rax
	je	.L747
	leaq	88(%rsp), %rbx
	.p2align 4,,10
	.p2align 3
.L745:
	movq	%rbx, %rsi
	movq	%r15, %rdi
.LEHB30:
	call	_ZNSt6vectorIiSaIiEEaSERKS1_
.LEHE30:
	addq	$24, %r15
	cmpq	%r15, %r12
	jne	.L745
.L747:
	movq	88(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L731
	call	_ZdlPv@PLT
	jmp	.L731
	.p2align 4,,10
	.p2align 3
.L809:
	movq	$-16, 16(%rsp)
.L770:
	movq	16(%rsp), %rdi
.LEHB31:
	call	_Znwm@PLT
.LEHE31:
	movq	%rax, 24(%rsp)
	jmp	.L771
	.p2align 4,,10
	.p2align 3
.L888:
	movq	%r13, %r12
	xorl	%ecx, %ecx
	jmp	.L735
	.p2align 4,,10
	.p2align 3
.L797:
	addq	$24, %r12
	cmpq	%r12, %r13
	jne	.L791
	jmp	.L798
	.p2align 4,,10
	.p2align 3
.L775:
	addq	%r10, %rcx
	movq	%rcx, 8(%rbx)
	addq	$24, %rbx
	decq	56(%rsp)
	jne	.L778
	jmp	.L776
	.p2align 4,,10
	.p2align 3
.L884:
	movq	%r13, %r10
	xorl	%ecx, %ecx
	jmp	.L773
	.p2align 4,,10
	.p2align 3
.L738:
	movq	%rbx, %r12
	subq	%rax, %r14
	jne	.L754
	jmp	.L748
	.p2align 4,,10
	.p2align 3
.L749:
	movabsq	$4611686018427387903, %rsi
	cmpq	%rsi, %rax
	ja	.L890
	movq	%r13, %rdi
.LEHB32:
	call	_Znwm@PLT
.LEHE32:
	movq	%rax, %rcx
.L750:
	addq	%rcx, %r13
	movq	%r13, 16(%r12)
	movq	%rcx, (%r12)
	movq	%rcx, 8(%r12)
	movq	96(%rsp), %rax
	movq	88(%rsp), %rsi
	movq	%rax, %r13
	subq	%rsi, %r13
	cmpq	%rsi, %rax
	je	.L879
	movq	%rcx, %rdi
	movq	%r13, %rdx
	call	memmove@PLT
	movq	%rax, %rcx
.L879:
	addq	%r13, %rcx
	movq	%rcx, 8(%r12)
	addq	$24, %r12
	decq	%r14
	je	.L748
.L754:
	movq	96(%rsp), %r13
	subq	88(%rsp), %r13
	movq	%r13, %rax
	sarq	$2, %rax
	movq	$0, (%r12)
	movq	$0, 8(%r12)
	movq	$0, 16(%r12)
	jne	.L749
	xorl	%ecx, %ecx
	jmp	.L750
	.p2align 4,,10
	.p2align 3
.L748:
	movq	64(%rsp), %rax
	movq	56(%rsp), %rsi
	movq	%r12, 8(%rax)
	movq	72(%rsp), %rax
	addq	%r12, %rsi
	cmpq	%rbx, %rax
	je	.L891
	.p2align 4,,10
	.p2align 3
.L763:
	movq	$0, 8(%r12)
	movq	$0, 16(%r12)
	movq	$0, (%r12)
	addq	$24, %rax
	addq	$24, %r12
	movq	-24(%rax), %rdx
	movq	%rdx, -24(%r12)
	movq	$0, -24(%rax)
	movq	-16(%rax), %rcx
	movq	-16(%r12), %rdx
	movq	%rcx, -16(%r12)
	movq	%rdx, -16(%rax)
	movq	-8(%rax), %rcx
	movq	-8(%r12), %rdx
	movq	%rcx, -8(%r12)
	movq	%rdx, -8(%rax)
	cmpq	%rax, %rbx
	jne	.L763
	movq	64(%rsp), %rax
	leaq	88(%rsp), %r12
	movq	%rsi, 8(%rax)
	.p2align 4,,10
	.p2align 3
.L765:
	movq	%r12, %rsi
	movq	%r15, %rdi
.LEHB33:
	call	_ZNSt6vectorIiSaIiEEaSERKS1_
.LEHE33:
	addq	$24, %r15
	cmpq	%rbx, %r15
	jne	.L765
	jmp	.L747
	.p2align 4,,10
	.p2align 3
.L886:
	vzeroupper
	jmp	.L796
	.p2align 4,,10
	.p2align 3
.L883:
	leaq	(%rdi,%rdi,2), %rax
	salq	$3, %rax
	movq	%rax, 16(%rsp)
	jmp	.L770
.L811:
	movq	%rbx, %rax
	jmp	.L792
.L891:
	movq	64(%rsp), %rax
	movq	%rsi, 8(%rax)
	jmp	.L747
.L885:
.LEHB34:
	call	_ZSt17__throw_bad_allocv@PLT
.LEHE34:
.L887:
	call	__stack_chk_fail@PLT
.L890:
.LEHB35:
	call	_ZSt17__throw_bad_allocv@PLT
.LEHE35:
.L889:
.LEHB36:
	call	_ZSt17__throw_bad_allocv@PLT
.L882:
	leaq	.LC8(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE36:
.L815:
	jmp	.L757
.L812:
	movq	%rax, %rbx
	jmp	.L762
.L817:
	jmp	.L781
.L757:
	movq	%rax, %rdi
	vzeroupper
	call	__cxa_begin_catch@PLT
.L760:
	cmpq	%r12, %rbx
	jne	.L892
.LEHB37:
	call	__cxa_rethrow@PLT
.LEHE37:
.L761:
	vzeroupper
	call	__cxa_end_catch@PLT
.L762:
	movq	88(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L875
	vzeroupper
	call	_ZdlPv@PLT
.L880:
	movq	%rbx, %rdi
.LEHB38:
	call	_Unwind_Resume@PLT
.LEHE38:
.L781:
	movq	%rax, %rdi
	vzeroupper
	call	__cxa_begin_catch@PLT
	movq	8(%rsp), %r12
.L784:
	cmpq	%rbx, %r12
	jne	.L893
.LEHB39:
	call	__cxa_rethrow@PLT
.LEHE39:
.L875:
	vzeroupper
	jmp	.L880
.L892:
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.L759
	call	_ZdlPv@PLT
.L759:
	addq	$24, %rbx
	jmp	.L760
.L893:
	movq	(%r12), %rdi
	testq	%rdi, %rdi
	je	.L783
	call	_ZdlPv@PLT
.L783:
	addq	$24, %r12
	jmp	.L784
.L814:
	movq	%rax, %rbx
	jmp	.L761
.L816:
	movq	%rax, %rbx
.L785:
	vzeroupper
	call	__cxa_end_catch@PLT
	movq	%rbx, %rdi
	call	__cxa_begin_catch@PLT
	cmpq	$0, 24(%rsp)
	je	.L894
	movq	24(%rsp), %rdi
	call	_ZdlPv@PLT
.L802:
.LEHB40:
	call	__cxa_rethrow@PLT
.LEHE40:
.L894:
	addq	(%rsp), %r14
	imulq	$24, %r14, %r14
	movq	8(%rsp), %rbx
.L804:
	cmpq	%rbx, %r14
	je	.L802
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.L803
	call	_ZdlPv@PLT
.L803:
	addq	$24, %rbx
	jmp	.L804
.L813:
	movq	%rax, %rbx
.L805:
	vzeroupper
	call	__cxa_end_catch@PLT
	jmp	.L880
	.cfi_endproc
.LFE3468:
	.section	.gcc_except_table
	.align 4
.LLSDA3468:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT3468-.LLSDATTD3468
.LLSDATTD3468:
	.byte	0x1
	.uleb128 .LLSDACSE3468-.LLSDACSB3468
.LLSDACSB3468:
	.uleb128 .LEHB28-.LFB3468
	.uleb128 .LEHE28-.LEHB28
	.uleb128 .L817-.LFB3468
	.uleb128 0x1
	.uleb128 .LEHB29-.LFB3468
	.uleb128 .LEHE29-.LEHB29
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB30-.LFB3468
	.uleb128 .LEHE30-.LEHB30
	.uleb128 .L812-.LFB3468
	.uleb128 0
	.uleb128 .LEHB31-.LFB3468
	.uleb128 .LEHE31-.LEHB31
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB32-.LFB3468
	.uleb128 .LEHE32-.LEHB32
	.uleb128 .L815-.LFB3468
	.uleb128 0x1
	.uleb128 .LEHB33-.LFB3468
	.uleb128 .LEHE33-.LEHB33
	.uleb128 .L812-.LFB3468
	.uleb128 0
	.uleb128 .LEHB34-.LFB3468
	.uleb128 .LEHE34-.LEHB34
	.uleb128 .L817-.LFB3468
	.uleb128 0x1
	.uleb128 .LEHB35-.LFB3468
	.uleb128 .LEHE35-.LEHB35
	.uleb128 .L815-.LFB3468
	.uleb128 0x1
	.uleb128 .LEHB36-.LFB3468
	.uleb128 .LEHE36-.LEHB36
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB37-.LFB3468
	.uleb128 .LEHE37-.LEHB37
	.uleb128 .L814-.LFB3468
	.uleb128 0
	.uleb128 .LEHB38-.LFB3468
	.uleb128 .LEHE38-.LEHB38
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB39-.LFB3468
	.uleb128 .LEHE39-.LEHB39
	.uleb128 .L816-.LFB3468
	.uleb128 0x3
	.uleb128 .LEHB40-.LFB3468
	.uleb128 .LEHE40-.LEHB40
	.uleb128 .L813-.LFB3468
	.uleb128 0
.LLSDACSE3468:
	.byte	0x1
	.byte	0
	.byte	0
	.byte	0x7d
	.align 4
	.long	0

.LLSDATT3468:
	.section	.text._ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_,"axG",@progbits,_ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_,comdat
	.size	_ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_, .-_ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_
	.section	.text.unlikely
	.align 2
.LCOLDB10:
	.text
.LHOTB10:
	.align 2
	.p2align 4,,15
	.globl	_ZN8KnnModel5SolveEv
	.type	_ZN8KnnModel5SolveEv, @function
_ZN8KnnModel5SolveEv:
.LFB2887:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2887
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rdi, %rbx
	subq	$40, %rsp
	.cfi_def_cfa_offset 80
	movslq	8(%rdi), %rdx
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	movq	$0, (%rsp)
	movq	$0, 8(%rsp)
	movq	$0, 16(%rsp)
	testq	%rdx, %rdx
	je	.L912
	movabsq	$4611686018427387903, %rax
	cmpq	%rax, %rdx
	ja	.L924
	leaq	0(,%rdx,4), %rbp
	movq	%rbp, %rdi
.LEHB41:
	call	_Znwm@PLT
	leaq	(%rax,%rbp), %r12
	movq	%rbp, %rdx
	movl	$255, %esi
	movq	%rax, %rdi
	movq	%rax, (%rsp)
	movq	%r12, 16(%rsp)
	call	memset@PLT
	movq	%rax, %rcx
.L896:
	movq	%r12, 8(%rsp)
	movq	56(%rbx), %r12
	movq	48(%rbx), %rsi
	movq	%r12, %rax
	subq	%rsi, %rax
	sarq	$3, %rax
	movabsq	$-6148914691236517205, %rdi
	imulq	%rdi, %rax
	movslq	(%rbx), %rdx
	cmpq	%rax, %rdx
	ja	.L925
	jb	.L926
.L899:
	testq	%rcx, %rcx
	je	.L904
	movq	%rcx, %rdi
	call	_ZdlPv@PLT
.L904:
	movq	%rbx, %rdi
	call	_ZN8KnnModel24PreCalculationOfDistanceEv
	movl	40(%rbx), %eax
	cmpl	$1, %eax
	je	.L905
	testl	%eax, %eax
	je	.L906
	cmpl	$2, %eax
	je	.L907
.L895:
	movq	24(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L927
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L907:
	.cfi_restore_state
	movq	%rbx, %rdi
	call	_ZN8KnnModel10_SolveHeapEv
.LEHE41:
	jmp	.L895
	.p2align 4,,10
	.p2align 3
.L926:
	leaq	(%rdx,%rdx,2), %rax
	leaq	(%rsi,%rax,8), %r13
	cmpq	%r13, %r12
	je	.L899
	movq	%r13, %rbp
	.p2align 4,,10
	.p2align 3
.L903:
	movq	0(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L900
	addq	$24, %rbp
	call	_ZdlPv@PLT
	cmpq	%rbp, %r12
	jne	.L903
.L901:
	movq	%r13, 56(%rbx)
	jmp	.L923
	.p2align 4,,10
	.p2align 3
.L925:
	movq	%rsp, %rcx
	subq	%rax, %rdx
	leaq	48(%rbx), %rdi
	movq	%r12, %rsi
.LEHB42:
	call	_ZNSt6vectorIS_IiSaIiEESaIS1_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS1_S3_EEmRKS1_
.LEHE42:
.L923:
	movq	(%rsp), %rcx
	jmp	.L899
	.p2align 4,,10
	.p2align 3
.L912:
	xorl	%ecx, %ecx
	xorl	%r12d, %r12d
	jmp	.L896
	.p2align 4,,10
	.p2align 3
.L906:
	movq	%rbx, %rdi
.LEHB43:
	call	_ZN8KnnModel11_SolveNaiveEv
	jmp	.L895
	.p2align 4,,10
	.p2align 3
.L905:
	movq	%rbx, %rdi
	call	_ZN8KnnModel12_SolveNaive2Ev
	jmp	.L895
	.p2align 4,,10
	.p2align 3
.L900:
	addq	$24, %rbp
	cmpq	%rbp, %r12
	jne	.L903
	jmp	.L901
.L927:
	call	__stack_chk_fail@PLT
.L924:
	call	_ZSt17__throw_bad_allocv@PLT
.LEHE43:
.L913:
	movq	%rax, %rbx
	jmp	.L909
	.section	.gcc_except_table
.LLSDA2887:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2887-.LLSDACSB2887
.LLSDACSB2887:
	.uleb128 .LEHB41-.LFB2887
	.uleb128 .LEHE41-.LEHB41
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB42-.LFB2887
	.uleb128 .LEHE42-.LEHB42
	.uleb128 .L913-.LFB2887
	.uleb128 0
	.uleb128 .LEHB43-.LFB2887
	.uleb128 .LEHE43-.LEHB43
	.uleb128 0
	.uleb128 0
.LLSDACSE2887:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC2887
	.type	_ZN8KnnModel5SolveEv.cold.186, @function
_ZN8KnnModel5SolveEv.cold.186:
.LFSB2887:
.L909:
	.cfi_def_cfa_offset 80
	.cfi_offset 3, -40
	.cfi_offset 6, -32
	.cfi_offset 12, -24
	.cfi_offset 13, -16
	movq	(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L920
	vzeroupper
	call	_ZdlPv@PLT
.L910:
	movq	%rbx, %rdi
.LEHB44:
	call	_Unwind_Resume@PLT
.LEHE44:
.L920:
	vzeroupper
	jmp	.L910
	.cfi_endproc
.LFE2887:
	.section	.gcc_except_table
.LLSDAC2887:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC2887-.LLSDACSBC2887
.LLSDACSBC2887:
	.uleb128 .LEHB44-.LCOLDB10
	.uleb128 .LEHE44-.LEHB44
	.uleb128 0
	.uleb128 0
.LLSDACSEC2887:
	.section	.text.unlikely
	.text
	.size	_ZN8KnnModel5SolveEv, .-_ZN8KnnModel5SolveEv
	.section	.text.unlikely
	.size	_ZN8KnnModel5SolveEv.cold.186, .-_ZN8KnnModel5SolveEv.cold.186
.LCOLDE10:
	.text
.LHOTE10:
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC11:
	.string	"GSE128223.inp"
.LC12:
	.string	"Start solving..."
.LC13:
	.string	"GSE128223.out"
	.section	.text.unlikely
.LCOLDB14:
	.section	.text.startup,"ax",@progbits
.LHOTB14:
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB2870:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2870
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	leaq	13+.LC11(%rip), %rdx
	leaq	-13(%rdx), %rsi
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	subq	$160, %rsp
	.cfi_def_cfa_offset 192
	leaq	112(%rsp), %rbp
	movq	%fs:40, %rax
	movq	%rax, 152(%rsp)
	xorl	%eax, %eax
	movq	%rbp, %rdi
	leaq	16(%rbp), %rax
	movq	$0, 32(%rsp)
	movq	$0, 40(%rsp)
	movq	$0, 48(%rsp)
	movq	$0, 64(%rsp)
	movq	$0, 72(%rsp)
	movq	$0, 80(%rsp)
	movq	$0, 88(%rsp)
	movq	$0, 96(%rsp)
	movq	$0, 104(%rsp)
	movq	%rax, 112(%rsp)
	leaq	16(%rsp), %r12
.LEHB45:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag.constprop.174
.LEHE45:
	leaq	16(%rsp), %r12
	movq	%rbp, %rsi
	movq	%r12, %rdi
.LEHB46:
	call	_ZN8KnnModel8ReadDataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.LEHE46:
	movq	112(%rsp), %rdi
	leaq	16(%rbp), %rax
	cmpq	%rax, %rdi
	je	.L929
	call	_ZdlPv@PLT
.L929:
	movl	$16, %edx
	leaq	.LC12(%rip), %rsi
	leaq	_ZSt4cout(%rip), %rdi
	movl	$2, 56(%rsp)
.LEHB47:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	leaq	_ZSt4cout(%rip), %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%r12, %rdi
	movq	%rax, %rbx
	call	_ZN8KnnModel5SolveEv
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	subq	%rbx, %rax
	movabsq	$2361183241434822607, %rdx
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	movq	%rcx, %rdi
	movq	%rdx, %rbx
	movabsq	$1237940039285380275, %rdx
	imulq	%rdx
	sarq	$63, %rdi
	sarq	$7, %rbx
	movq	%rdx, %rsi
	sarq	$26, %rsi
	subq	%rdi, %rbx
	subq	%rdi, %rsi
	leaq	_ZSt4cout(%rip), %rdi
	call	_ZNSo9_M_insertIlEERSoT_@PLT
	leaq	14(%rsp), %rsi
	movl	$1, %edx
	movq	%rax, %rdi
	movb	$46, 14(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	movabsq	$4835703278458516699, %rdx
	movq	%rbx, %rax
	imulq	%rdx
	movq	%rbx, %rax
	sarq	$63, %rax
	sarq	$18, %rdx
	subq	%rax, %rdx
	imulq	$1000000, %rdx, %rdx
	subq	%rdx, %rbx
	movq	%rbx, %rsi
	call	_ZNSo9_M_insertIlEERSoT_@PLT
	leaq	15(%rsp), %rsi
	movl	$1, %edx
	movq	%rax, %rdi
	movb	$115, 15(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rax, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@PLT
	leaq	13+.LC13(%rip), %rdx
	leaq	16(%rbp), %rax
	leaq	-13(%rdx), %rsi
	movq	%rbp, %rdi
	movq	%rax, 112(%rsp)
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag.constprop.174
.LEHE47:
	movq	%rbp, %rsi
	movq	%r12, %rdi
.LEHB48:
	call	_ZN8KnnModel6OutputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.LEHE48:
	movq	112(%rsp), %rdi
	addq	$16, %rbp
	cmpq	%rbp, %rdi
	je	.L930
	call	_ZdlPv@PLT
.L930:
	movq	%r12, %rdi
	call	_ZN8KnnModelD1Ev
	xorl	%eax, %eax
	movq	152(%rsp), %rcx
	xorq	%fs:40, %rcx
	jne	.L945
	addq	$160, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L945:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
.L939:
	movq	%rax, %rbx
	jmp	.L934
.L937:
	movq	%rax, %rbx
	jmp	.L931
.L938:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L933
	.section	.gcc_except_table
.LLSDA2870:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2870-.LLSDACSB2870
.LLSDACSB2870:
	.uleb128 .LEHB45-.LFB2870
	.uleb128 .LEHE45-.LEHB45
	.uleb128 .L938-.LFB2870
	.uleb128 0
	.uleb128 .LEHB46-.LFB2870
	.uleb128 .LEHE46-.LEHB46
	.uleb128 .L937-.LFB2870
	.uleb128 0
	.uleb128 .LEHB47-.LFB2870
	.uleb128 .LEHE47-.LEHB47
	.uleb128 .L938-.LFB2870
	.uleb128 0
	.uleb128 .LEHB48-.LFB2870
	.uleb128 .LEHE48-.LEHB48
	.uleb128 .L939-.LFB2870
	.uleb128 0
.LLSDACSE2870:
	.section	.text.startup
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC2870
	.type	main.cold.187, @function
main.cold.187:
.LFSB2870:
.L934:
	.cfi_def_cfa_offset 192
	.cfi_offset 3, -32
	.cfi_offset 6, -24
	.cfi_offset 12, -16
	movq	112(%rsp), %rdi
	addq	$16, %rbp
	cmpq	%rbp, %rdi
	je	.L942
	vzeroupper
	call	_ZdlPv@PLT
.L933:
	movq	%r12, %rdi
	call	_ZN8KnnModelD1Ev
	movq	%rbx, %rdi
.LEHB49:
	call	_Unwind_Resume@PLT
.LEHE49:
.L931:
	movq	112(%rsp), %rdi
	addq	$16, %rbp
	cmpq	%rbp, %rdi
	je	.L941
	vzeroupper
	call	_ZdlPv@PLT
	jmp	.L933
.L941:
	vzeroupper
	jmp	.L933
.L942:
	vzeroupper
	jmp	.L933
	.cfi_endproc
.LFE2870:
	.section	.gcc_except_table
.LLSDAC2870:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC2870-.LLSDACSBC2870
.LLSDACSBC2870:
	.uleb128 .LEHB49-.LCOLDB14
	.uleb128 .LEHE49-.LEHB49
	.uleb128 0
	.uleb128 0
.LLSDACSEC2870:
	.section	.text.unlikely
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold.187, .-main.cold.187
.LCOLDE14:
	.section	.text.startup
.LHOTE14:
	.p2align 4,,15
	.type	_GLOBAL__sub_I_main, @function
_GLOBAL__sub_I_main:
.LFB4073:
	.cfi_startproc
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	leaq	_ZStL8__ioinit(%rip), %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	__dso_handle(%rip), %rdx
	leaq	_ZStL8__ioinit(%rip), %rsi
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE4073:
	.size	_GLOBAL__sub_I_main, .-_GLOBAL__sub_I_main
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_main
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC4:
	.long	0
	.long	-1074790400
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 8.4.0-1ubuntu1~18.04) 8.4.0"
	.section	.note.GNU-stack,"",@progbits
